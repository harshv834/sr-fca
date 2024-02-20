import torch
import numpy as np
from tqdm import tqdm
import os
from collections import OrderedDict
from sklearn.cluster import KMeans


from cifar_trainers import  ClusterTrainer, ClientTrainer
from cifar_utils import create_argparse, create_config, calc_acc
from cifar_dataset import create_client_loaders

parser = create_argparse()




def vectorize_model_wts(model):
    """Flatten all model weights into single vector

    Args:
        model (nn.Module): Model whose weights need to be flattened

    Returns:
        np.ndarray: 1D vector of flattened model weigts.
    """
    model_wts = model.state_dict()
    wt_list = [
        wt.numpy().flatten()
        for wt in list(model_wts.values())
    ]
    wt_list = [wt for wt in wt_list if np.issubdtype(wt.dtype, np.integer) or np.issubdtype(wt.dtype, np.floating)]
    vectorized_wts = np.hstack(wt_list)
    return vectorized_wts


def unvectorize_model_wts(flat_wts, model):
    """Convert flattened model weights to an ordered state dict of the model

    Args:
        flat_wts (np.ndarray): 1D array with the flattened weights
        model (torch.nn.Module): Model whose state dict format we need to adhere to

    Returns:
        OrderedDict: Format flat_wts according to state dict of model
    """
    model_wts = model.state_dict()
    model_wts_to_update = OrderedDict(
        {
            key: val
            for key, val in model_wts.items()
            if np.issubdtype(val.numpy().dtype, np.integer) or np.issubdtype(val.numpy().dtype, np.floating)
        }
    )
    flat_tensor_len = [val.flatten().shape[0] for val in model_wts_to_update.values()]
    start_count = 0
    for i, (key, val) in enumerate(model_wts_to_update.items()):
        end_count = start_count + flat_tensor_len[i]
        flat_tensor = flat_wts[start_count:end_count]
        model_wts_to_update[key] = torch.tensor(
            flat_tensor.reshape(val.shape), dtype=val.dtype
        )
    model_wts_to_update.update(model_wts)
    return model_wts_to_update



args = parser.parse_args()
config=create_config(args)
client_loaders = create_client_loaders(config)
init_path = os.path.join(config["results_dir"], "init")
client_trainers = [
    ClientTrainer(
        config, os.path.join(init_path, "node_{}".format(i)), i
    )
    for i in range(config["num_clients"])
]

## If saved models present in init then start SR_FCA from there
if args.from_init:
    ## Get the path from init
    for i in tqdm(range(config["num_clients"])):
        ## Load saved model weights
        client_trainers[i].load_saved_weights()
    # self.init_metrics = torch.load(os.path.join(init_path, "metrics.pth"))
else:
    for i in tqdm(range(config["num_clients"])):
        client_trainers[i].train(client_loaders[i])
        
# import ipdb;ipdb.set_trace()
kmeans_path = os.path.join(config["results_dir"], "kmeans")
kmeans_metrics = []

client_model_wts = np.vstack(
    [
        vectorize_model_wts(trainer.model)
        for trainer in client_trainers
    ]
)
kmeans_model = KMeans(
    n_clusters=config["num_clusters"],
    random_state=config["seed"],
    init="k-means++",
)
kmeans_model.fit(client_model_wts)
cluster_map = {}
cluster_trainers = {}
kmeans_metrics = []
# import ipdb;ipdb.set_trace()
test_acc = 0.0

for i in range(config["num_clusters"]):
    cluster_clients = np.where(kmeans_model.labels_ == i)[0].tolist()
    if len(cluster_clients) > 0:
        cluster_test_acc = 0.0
        cluster_map[i] = cluster_clients
        cluster_center_wts = unvectorize_model_wts(
            kmeans_model.cluster_centers_[i], client_trainers[0].model
        )
        cluster_trainer = ClusterTrainer(config, os.path.join(kmeans_path, "cluster_{}".format(i)), i)
        cluster_trainer.model.load_state_dict(cluster_center_wts)
        cluster_trainer.model.to(memory_format=torch.channels_last).cuda()
        for client_idx in cluster_clients:
            test_acc += calc_acc(cluster_trainer.model,config["device"], client_loaders[client_idx], train=False)
        cluster_trainers[i] = cluster_trainer
test_acc = test_acc/16
torch.save({"test_acc" : test_acc}, os.path.join(kmeans_path, "metrics.pth"))
torch.save(cluster_map, os.path.join(kmeans_path,  "cluster_map.pth"))
print("Final Test acc : {}".format(test_acc))


