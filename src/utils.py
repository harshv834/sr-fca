import argparse
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict, OrderedDict
from shutil import rmtree

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.cuda.amp import autocast
from math import ceil

import torch.nn as nn


def args_getter():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="random seed for this experiment"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        # required=True,
        default="synthetic",
        choices=[
            "synthetic",
            "rot_mnist",
            "inv_mnist",
            "rot_cifar10",
            "shakespeare",
            "femnist",
            "rot_cifar10_ftrs",
        ],
        help="Dataset to use for this run",
    )
    parser.add_argument(
        "-c",
        "--clustering",
        choices=["sr_fca", "ifca", "cfl", "oneshot_kmeans", "fedavg", "soft_ifca", "oneshot_ifca", "feddrift", "sr_fca_merge_refine"],
        # required=True,
        default="sr_fca",
        help="Clustering algo to use for this run, if all is specified run all algorithms and compare",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        required=False,
        help="Number of clusters for IFCA/CFL/Oneshot_KMeans",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=False,
        help="Number of samples for tuning",
    )

    parser.add_argument(
        "--from_init",
        action=argparse.BooleanOptionalAction  
    )
    # parser.add_argument(
    #     "--dist-metric",
    #     choices=["euclidean", "cross_entropy"],
    #     # required=True,
    #     default="euclidean",
    #     help="Distance metric to use when comparing models",
    # )
    args = parser.parse_args()
    args = vars(args)
    args = {key: val for key, val in args.items() if val is not None}
    return args
    # ## This is set by hyperparameter config. Do we need a single config file with everything
    # parser.add_argument("-b", "--base-optimizer", choices=["adam", "sgd", "arg"])


def read_algo_config(data_config, tune=False):
    path = os.path.join(
        "configs",
        "clustering",
        data_config["clustering"],
        data_config["dataset"]["name"],
        "best_hp.yaml",
    )
    exists = os.path.exists(path)
    if tune:
        if exists:
            rmtree(path)

        # config = suggest_config(
        #     data_config["dataset"]["name"], data_config["clustering"]
        # )
    else:

        assert exists, "Algorithm config does not exist for path {}".format(path)

        with open(path, "r") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as err:
                print(err)
    ## Works only on python >=3.9.0

    config = config | data_config

    if config["dataset"]["name"] == "synthetic":
        config["model"]["params"] = {
            "dimension": config["dataset"]["dimension"],
            "scale": config["dataset"]["scale"],
        }
    return config


LOSS_DICT = {"cross_entropy": torch.nn.CrossEntropyLoss(), "mse": torch.nn.MSELoss()}


def compute_metric(
    model, client_data, train=True, loss=None, device=None, lstm_flag=False, return_list=False
):
    
    loader = client_data.trainloader if train else client_data.testloader
    model.eval()

    if device is not None:
        # model = model.to(memory_format=torch.channels_last)
        model = model.cuda()
    if lstm_flag:
        batch_size, hidden = None, None
    with torch.no_grad():
        if return_list:
            total_list = []
        else:
            total, total_num = 0.0, 0.0
        for X, Y in loader:
            if lstm_flag:
                if batch_size is None or X.shape[0] != batch_size:
                    batch_size = X.shape[0]
                    hidden = model.zero_state(batch_size)
            if device is not None:
                X, Y = X.to(device), Y.to(device)
                if lstm_flag:
                    hidden = (hidden[0].to(device), hidden[1].to(device))

            with autocast():
                if lstm_flag:
                    out, hidden = model(X, hidden)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                else:
                    out = model(X)  # Test-time augmentation

                if loss is not None:
                    if return_list:
                    ## This should copy old behavior hopefully
                        total_list.append(loss(out, Y).detach().cpu().numpy())
                    else:
                        total += loss(out, Y).item()
                else:
                    total += out.argmax(1).eq(Y).sum().cpu().item()
                if not return_list:
                    total_num += Y.shape[0]
    if return_list:
        output = np.hstack(total_list)
    else:
        output = total/total_num

    return output


def get_optimizer(model_parameters, config):
    optimizer_name = config["optimizer"]["name"]
    optimizer_params = config["optimizer"]["params"]
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model_parameters, **optimizer_params)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model_parameters, **optimizer_params)
    else:
        raise ValueError("Invalid optimizer name {}".format(optimizer_name))
    return optimizer


def get_lr_scheduler(config, optimizer, local_iter, round_id):
    cond_1 = config["dataset"]["name"] == "rot_cifar10"
    cond_2 = config["clustering"] == "sr_fca"
    if cond_1:
        iters_per_epoch = 50000 // int(config["batch"]["train"])
        if cond_2:
            epochs = config["init"]["iterations"] // iters_per_epoch
        else:
            epochs = (config["rounds"] * config["local_iter"]) // iters_per_epoch
        lr_schedule = np.interp(
            np.arange((epochs + 1) * iters_per_epoch),
            [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
            [0, 1, 0],
        )
        if round_id is not None:
            if type(round_id) == tuple:
                first_iter = round_id[0] * local_iter
                last_iter = min(
                    (round_id[1] + 1) * local_iter + 1, lr_schedule.shape[0]
                )
            else:
                first_iter = round_id * local_iter
                last_iter = min((round_id + 1) * local_iter + 1, lr_schedule.shape[0])
            lr_schedule = lr_schedule[first_iter:last_iter]
    # elif config["dataset"]["name"] == "rot_cifar10_ftrs":
    #     return optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2)
    else:
        if cond_2:
            lr_schedule = np.ones(config["init"]["iterations"] + 1)
        else:
            lr_schedule = np.ones(max(config["rounds"], local_iter) + 1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    return scheduler


def avg_metrics(metrics_list, min_tot_count=1):
    '''
    min_tot_count is minimum value of tot_count
    '''
    avg_metric_dict = {}

    metric_keys = metrics_list[0][1].keys()
    for key in metric_keys:
        avg_metric_dict[key] = {}
        for metric_name in metrics_list[0][1][key].keys():
            avg_metric_dict[key][metric_name] = 0.0
    
    tot_count = 0
    for (count, metric_dict) in metrics_list:
        tot_count += count
        for key in metric_keys:
            for metric_name in metric_dict[key].keys():
                avg_metric_dict[key][metric_name] += (
                    count * metric_dict[key][metric_name]
                )
    for key in metric_keys:
        for metric_name in metrics_list[0][1][key].keys():
            avg_metric_dict[key][metric_name] = (
                avg_metric_dict[key][metric_name] / max(tot_count, min_tot_count)
            )
    return avg_metric_dict


def euclidean_dist(w1, w2):
    dist = 0.0
    for key in w1.keys():
        dist += np.linalg.norm(w1[key] - w2[key]) ** 2
    return np.sqrt(dist)


def compute_dist(trainer_1, trainer_2, client_1, client_2, dist_metric):
    if dist_metric == "euclidean":
        return euclidean_dist(trainer_1.get_model_wts(), trainer_2.get_model_wts())
    elif dist_metric == "cross_entropy":
        trainer_1_client_2 = 0.0
        for client in client_2:
            trainer_1_client_2 += trainer_1.compute_loss(client)
        trainer_1_client_2 = trainer_1_client_2 / len(client_2)
        trainer_2_client_1 = 0.0
        for client in client_1:
            trainer_2_client_1 += trainer_2.compute_loss(client)
        trainer_2_client_1 = trainer_2_client_1 / len(client_1)
        return (trainer_1_client_2 + trainer_2_client_1) / 2
    else:
        raise ValueError(
            "{} is not a valid distance metric. Implemented distance metrics are euclidean and cross_entropy".format(
                dist_metric
            )
        )


def correlation_clustering(client_graph, size_threshold):
    clustering = []
    while len(client_graph.nodes) > 0:
        cluster = []
        new_cluster_pivot = random.sample(list(client_graph.nodes), 1)[0]
        cluster.append(new_cluster_pivot)
        neighbors = client_graph[new_cluster_pivot].copy()
        for node in neighbors:
            cluster.append(node)
            client_graph.remove_node(node)
        client_graph.remove_node(new_cluster_pivot)
        clustering.append(cluster)
    clusters = [cluster for cluster in clustering if len(cluster) > size_threshold]
    cluster_map = {i: clusters[i] for i in range(len(clusters))}
    return cluster_map


def read_data_config(args_dict):
    path = os.path.join("configs", "experiment", args_dict["dataset"] + ".yaml")
    assert os.path.exists(path), "Dataset config does exist for path {}".format(path)
    config = {}
    ## Config loaded from yaml file for the dataset and merged with the arguments from argparse
    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as err:
            print(err)
    # this updates args_dict by config with replacement
    config = args_dict | config
    config["path"] = {
        "data": os.path.join(config["path"], "seed_{}".format(config["seed"]), "data"),
        "results": os.path.join(
            config["path"],
            "seed_{}".format(config["seed"]),
            "results",
            args_dict["clustering"],
        ),
    }
    ## Make directories for data and results
    os.makedirs(config["path"]["data"], exist_ok=True)
    os.makedirs(config["path"]["results"], exist_ok=True)

    return config


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ## What does this do ?
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# @ray.remote(num_gpus=1)
def get_device(config, i, cluster=False):
    num_devices = torch.cuda.device_count()
    if num_devices >= 1:
        # return "cuda:{}".format(i%num_devices)
        # if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        #     CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
        #     return "cuda:" + os.environ["CUDA_VISIBLE_DEVICES"]
        # else:
        return "cuda:0"
    else:
        raise ValueError(
            "Current implementation can handle only >=1 GPU. {} GPUs were provided".format(
                num_devices
            )
        )


def check_nan(metrics):
    for key in metrics.keys():
        for val in metrics[key].values():
            if np.isnan(np.array(val)).any() or np.isinf(np.array(val)).any():
                return True

    return False


def get_search_space(config):
    algo_config_path = os.path.join(
        "configs",
        "clustering",
        config["clustering"],
        config["dataset"]["name"],
    )
    hp_search_path = os.path.join(algo_config_path, "hp_config.py")
    spec = importlib.util.spec_from_file_location("module.name", hp_search_path)
    config_file = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = config_file
    spec.loader.exec_module(config_file)
    best_hp_path = os.path.join(algo_config_path, "best_hp.yaml")
    return best_hp_path, lambda trial: config_file.get_hp_config(trial)


def wt_dict_diff(wt_1, wt_2):
    assert wt_1.keys() == wt_2.keys(), "Both weight dicts have different keys"
    diff_dict = {}
    for key in wt_1.keys():
        diff_dict[key] = convert_to_cpu(wt_1[key]) - convert_to_cpu(wt_2[key])
    return diff_dict

def wt_dict_mean(wts_coeff_list):
    # Input should be [(coeff_1, wt_1), (coeff_2, wt_2), ..]
    # Output is (coeff_1 * wt_1 + coeff_2 * wt_2 + ..)/(coeff_1 + coeff_2 + ..)
    
    coeff_list, wts_list = list(zip(*wts_coeff_list))
    coeff_arr = torch.tensor(coeff_list).float()
    first_wt = wts_list[0]
    for i, wt in enumerate(wts_list[1:]):
        assert first_wt.keys() == wt.keys(), f"First weight and weight {i} have different keys"
    
    mean_dict = {}    
    for key in first_wt.keys():
        mean_dict[key] = (torch.stack([wt[key] for wt in wts_list], dim=-1).cpu().float() @ coeff_arr)/ coeff_arr.sum()
        assert mean_dict[key].shape == first_wt[key].shape
        
    return mean_dict


def wt_dict_norm(wt):
    norm = 0
    for val in wt.values():
        norm += np.linalg.norm(convert_to_cpu(val)) ** 2
    return np.sqrt(norm)


def wt_dict_dot(wt_1, wt_2):
    assert wt_1.keys() == wt_2.keys(), "Both weight dicts have different keys"
    dot = 0.0
    for key in wt_1.keys():
        dot += np.dot(
            convert_to_cpu(wt_1[key]).reshape(-1), convert_to_cpu(wt_2[key]).reshape(-1)
        )
    return dot


def convert_to_cpu(val):
    val_arr = val
    if type(val) != np.ndarray:
        if val.device != "cpu":
            val_arr = val_arr.cpu()
        val_arr = val_arr.cpu()
    return val_arr


def compute_alpha_max(alpha_mat, partitions, client_idx):
    keys = list(partitions.keys())
    return alpha_mat[[client_idx.index(el) for el in partitions[keys[0]]], :][
        :, [client_idx.index(el) for el in partitions[keys[1]]]
    ].max()


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        data.update(cdata["user_data"])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def tune_config_update(config):
    if config["clustering"] == "sr_fca":
        config["refine"]["rounds"] = ceil(
            int(config["init"]["iterations"]) / int(config["refine"]["local_iter"])
        )

    elif config["clustering"] in ["ifca", "cfl", "fedavg"]:
        config["rounds"] = ceil(int(config["iterations"]) / int(config["local_iter"]))
    else:
        raise NotImplementedError(
            "Not implemented clustering {}".format(config["clustering"])
        )

    return config


def set_weights(name, model, path):
    model_wt = torch.load(path)

    # new_wts = OrderedDict()
    if name == "femnist":
        model.load_state_dict(model_wt)

        # new_wts["fc2.weight"] = torch.Tensor(model_wt["dense_1/kernel"]).t()
        # new_wts["fc2.bias"] = torch.Tensor(model_wt["dense_1/bias"])
        # new_wts["fc1.weight"] = torch.Tensor(model_wt["dense/kernel"]).t()
        # new_wts["fc1.bias"] = torch.Tensor(model_wt["dense/bias"])
        # new_wts["conv1.weight"] = torch.Tensor(model_wt["conv2d/kernel"]).permute(
        #     3, 2, 0, 1
        # )
        # new_wts["conv2.weight"] = torch.Tensor(model_wt["conv2d_1/kernel"]).permute(
        #     3, 2, 0, 1
        # )
        # new_wts["conv1.bias"] = torch.Tensor(model_wt["conv2d/bias"])
        # new_wts["conv2.bias"] = torch.Tensor(model_wt["conv2d_1/bias"])
        # model.load_state_dict(new_wts)
        # model.conv1.weight.requires_grad = False
        # model.conv2.weight.requires_grad = False
        # model.fc1.weight.requires_grad = True
        # model.fc2.weight.requires_grad = True
        # model.conv1.bias.requires_grad = False
        # model.conv2.bias.requires_grad = False
        # model.fc1.bias.requires_grad = True
        # model.fc2.bias.requires_grad = True
    elif name == "rot_cifar10":
        model.model.load_state_dict(model_wt)
        children = list(model.model.children())
        for child_id in range(len(children)):
            for name, param in children[child_id].named_parameters():
                if param.requires_grad:
                    param.requires_grad = child_id >= len(children) - 2
        model.model = nn.Sequential(*children)
    return model


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
    model_wts_to_update = model_wts | model_wts_to_update
    return model_wts_to_update

# def compute_misclustering(path, num_clusters, num_clients):
#     cluster_map  = torch.load(path)
#     client_idx = range(num_clients)
#     ## Define true clustering
#     true_clustering = {i : [j for j in client_idx if j% num_clusters == i] for i in range(num_clusters)}
    
#     ## Sort clusters based on size
#     cluster_idx = [(len(item), key) for key, item in cluster_map.items()]
#     sorted_cluster_idx = sorted(cluster_idx)
#     sorted_cluster_idx = [a[1] for a in cluster_idx]

#     ## Obtain best match possible
#     num_misclustered = 0.0
#     for idx in sorted_cluster_idx:
#         curr_closest_cluster = 
#         cluster = set(cluster_map[idx])
#         for true_cluster in true_clustering
#     return sorted_cluster_idx