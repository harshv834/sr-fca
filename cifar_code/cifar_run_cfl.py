import torch
import numpy as np
import os
from cifar_trainers import GlobalTrainer
from cifar_utils import create_argparse, create_config, wt_dict_norm, compute_alpha_max, wt_dict_dot, calc_acc
from cifar_dataset import create_client_loaders
import itertools
parser = create_argparse()
parser.add_argument(
    "--gamma_max", type=float, default = 0.5, help="gamma max for cfl"
)
parser.add_argument("--client_threshold", type=float, default = 0.1, help = "client threshold for cfl")
parser.add_argument("--stop_threshold", type=float, default = 0.1, help = "stop threshold for cfl")




cluster_map = {}
cluster_trainers = {}
cluster_metrics = {}
cluster_idx_to_train_queue = []







def cfl_single_node(config, client_dict, cluster_id):
    ## Train a model for the cluster
    cluster_save_dir = os.path.join(
        config["cluster_path"], "cluster_{}".format(cluster_id)
    )

    cluster_trainer = GlobalTrainer(
        config, cluster_save_dir
    )
    cluster_metrics[cluster_id] = cluster_trainer.train(
        client_data_list=list(client_dict.values()),
    )
    global cluster_trainers 
    global cluster_map
    global cluster_idx_to_train_queue

    cluster_trainers[cluster_id] = cluster_trainer

    ## Split cluster into two parts
    if (
        len(cluster_map[cluster_id]) == 1
        or len(cluster_map) == config["num_clusters"]
    ):
        return
    else:
        ## Compute alpha for every client pair in cluster

        alpha_mat, max_loss_client = compute_alpha_mat(cluster_id)
        if np.isnan(alpha_mat).any() or np.isinf(alpha_mat).any():
            print(
                "Nan or inf occurred in alpha for cluster : {}".format(cluster_id)
            )
        ## Obtain optimal bipartitioning to maximize
        partitions = optimal_bipartitioning(cluster_id, alpha_mat)

        ## Obtain max alpha between two partitions
        alpha_max_cross = compute_alpha_max(
            alpha_mat, partitions, cluster_map[cluster_id]
        )
        if (
            max_loss_client >= config["client_threshold"]
            and np.sqrt((1 - alpha_max_cross) / 2) > config["gamma_max"]
        ) or True:
            _ = cluster_map.pop(cluster_id)
            _ = cluster_trainers.pop(cluster_id)
            for key, val in partitions.items():
                cluster_map[key] = val
                cluster_idx_to_train_queue.append(key)
        return

def compute_alpha_mat(cluster_id):
    """Compute alpha matrix which is cosine similarity of loss gradient of different clients at optima for the cluster.

    Args:
        cluster_id (_type_): _description_

    Returns:
        _type_: final alpha matric
    """

    ## Cosine similarity is 1 if the two clients are same, so start
    ## with an identity matrix
    client_idx = cluster_map[cluster_id]
    alpha_mat = np.diag(np.ones(len(client_idx)))
    ## Compute weight differences/loss gradient at optima for clients in given cluster
    client_wt_diff = cluster_trainers[cluster_id].client_wt_diff 
    wt_diff_norms = {i: wt_dict_norm(client_wt_diff[i]) for i in range(len(client_idx))}
    for (i, j) in itertools.combinations(range(len(client_idx)), 2):
        if (
            wt_diff_norms[client_idx[i]] < 1e-10
            or wt_diff_norms[client_idx[j]] < 1e-10
        ):
            alpha_mat[i][j] = 0
        else:
            dot = wt_dict_dot(
                client_wt_diff[client_idx[i]], client_wt_diff[client_idx[j]]
            )
            alpha_mat[i][j] = dot / (
                wt_diff_norms[client_idx[i]] * wt_diff_norms[client_idx[j]]
            )
        alpha_mat[j][i] = alpha_mat[i][j]
    max_loss_client = max(wt_diff_norms.values())
    return alpha_mat, max_loss_client

def optimal_bipartitioning(cluster_id, alpha_mat):
    client_idx = cluster_map[cluster_id]
    num_clients = len(client_idx)
    alpha_flat = alpha_mat.flatten()
    sorted_idx = (-1 * alpha_flat).argsort()
    C = {i: set([i]) for i in client_idx}
    cluster_list = list(C.keys())
    for i in range(num_clients**2):
        i_1 = client_idx[sorted_idx[i] // num_clients]
        i_2 = client_idx[sorted_idx[i] % num_clients]
        c_temp = set([])
        j_min = max(cluster_list)
        for j in cluster_list:
            if i_1 in C[j] or i_2 in C[j]:
                j_min = min(j, j_min)
                c_temp = c_temp.union(C[j])
                C[j] = set()
        C[j_min] = c_temp
        cluster_list = []
        for key in C.keys():
            if len(C[key]) > 0:
                cluster_list.append(key)

        C = {j: C[j] for j in cluster_list}
        if len(cluster_list) == 2:
            partition_1_id = (cluster_id + 1) * 2
            partition_2_id = (cluster_id + 1) * 2 + 1
            return {
                partition_1_id: [client_id for client_id in C[cluster_list[0]]],
                partition_2_id: [client_id for client_id in C[cluster_list[1]]],
            }

    # def cluster(self, experiment):
    #     """Main method to create clusters of clients

    #     Args:
    #         experiment (dict): Dict of client data used for the experiment

    #     Raises:
    #         ValueError: When Nan or inf appears in metrics

    #     Returns:
    #         dict: Metrics of trained cluster federated learning 
    #     """
    #     ### Initialize the client dict and put all clients inside the first cluster which has cluster_id 0
    #     self.config["time"]["tcluster"] = time()

    #     client_dict = experiment.client_dict
    #     init_cluster_id = 0
    #     self.cluster_map = {init_cluster_id: list(range(self.config["num_clients"]))}

    #     ## Add this cluster_id to a FIFO queue which contains cluster_idx to train next
    #     self.cluster_idx_to_train.append(init_cluster_id)

    #     ## While required number of clusters haven't been trained, perform CFL on a new cluster id
    #     while len(self.cluster_trainers.keys()) < self.config["num_clusters"]:
    #         ## Put cluster_idx to train in a queue and pop the queue and train each cluster.
    #         if len(self.cluster_idx_to_train) > 0:
    #             cluster_idx_to_train = self.cluster_idx_to_train.pop(0)
    #             client_dict_to_train = {
    #                 client_idx: client_dict[client_idx]
    #                 for client_idx in self.cluster_map[cluster_idx_to_train]
    #             }
    #             self.cfl_single_node(client_dict_to_train, cluster_idx_to_train)
    #         else:
    #             break
    #     ## Among the final clusters which remain,
    #     self.metrics = []
    #     for cluster_id in self.cluster_map.keys():
    #         self.cluster_trainers[cluster_id].client_idx = self.cluster_map[cluster_id]
    #         metrics = self.cluster_trainers[cluster_id].compute_metrics(client_dict)
    #         if check_nan(metrics):
    #             raise ValueError("Nan or inf occurred in metrics")
    #         self.metrics.append((len(self.cluster_map[cluster_id]), metrics))
    #     self.metrics = avg_metrics(self.metrics)
    #     torch.save(self.metrics, os.path.join(self.config["path"]["results"], "metrics.pth"))
    #     torch.save(self.cluster_map, os.path.join(self.config["path"]["results"], "cluster_map.pth"))
    #     return self.metrics


def main(args):

    config = create_config(args)
    client_loaders = create_client_loaders(config)

    # rounds = 2400 // 10
    # local_iter = 10
    # config["iterations"] = 1
    
    
    config["cluster_path"] = os.path.join(config["results_dir"], "cfl", "clusters")

    global cluster_map 
    global cluster_trainers
    global cluster_metrics
    global cluster_idx_to_train_queue

    
    
    init_cluster_id = 0
    cluster_map = {init_cluster_id: list(range(config["num_clients"]))}

    ## Add this cluster_id to a FIFO queue which contains cluster_idx to train next
    cluster_idx_to_train_queue.append(init_cluster_id)

    ## While required number of clusters haven't been trained, perform CFL on a new cluster id
   
    while len(cluster_trainers.keys()) < config["num_clusters"]:
        ## Put cluster_idx to train in a queue and pop the queue and train each cluster.
        if len(cluster_idx_to_train_queue) > 0:
            cluster_idx_to_train = cluster_idx_to_train_queue.pop(0)
            client_dict_to_train = {
                client_idx: client_loaders[client_idx]
                for client_idx in cluster_map[cluster_idx_to_train]
            }
            cfl_single_node(config, client_dict_to_train, cluster_idx_to_train)
        else:
            break
    ## Among the final clusters which remain,
    test_acc = 0.0
    for cluster_id in cluster_map.keys():
        cluster_test_acc = 0.0
        for client_idx in cluster_map[cluster_id]:
            cluster_test_acc += calc_acc(cluster_trainers[cluster_id].model, config["device"], client_loaders[client_idx], train=False)
        test_acc += cluster_test_acc
    test_acc  = test_acc/16
    torch.save({"test_acc" : test_acc}, os.path.join(config["results_dir"], "cfl", "metrics.pth"))
    torch.save(cluster_map, os.path.join(config["results_dir"],"cfl",  "cluster_map.pth"))
    print("Test Acc : {}".format(test_acc))
    return test_acc
    
    
    
    
    # ## Initial cluster_map
    # all_clients = list(range(16))
    # np.random.shuffle(all_clients)
    # cluster_map = {0: all_clients[:8], 1: all_clients[8:]}


    # # cluster_map = {0: [0, 2, 4, 6, 8, 10, 12, 14], 1: [1, 3, 5, 7, 9, 11, 13, 15]}


    # client_loaders = np.array(client_loaders)
    # ifca_trainers = [
    #     GlobalTrainer(config, os.path.join(config["results_dir"], "ifca", "cluster_0")),
    #     GlobalTrainer(
    #         config,
    #         os.path.join(
    #             config["results_dir"],
    #             "ifca",
    #             "cluster_1",
    #         ),
    #     ),
    # ]
    # iters_per_epoch = 50000 // int(
    #     config["train_batch"] * config["total_num_clients_per_cluster"]
    # )
    # epochs = 2400 // iters_per_epoch
    # lr_schedule = np.interp(
    #     np.arange((epochs + 1) * iters_per_epoch),
    #     [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
    #     [0, 1, 0],
    # )
    # metrics = {0:{"train_loss" : [], "test_acc":  []}, 1:{"train_loss" : [], "test_acc" : []}}
    # for round_id in tqdm(range(rounds)):
    #     for i in range(2):
    #         if len(cluster_map[i]) > 0:
    #             train_loss_list, test_acc_list = ifca_trainers[i].train(
    #                 client_loaders[cluster_map[i]],
    #                 lr_schedule[
    #                     round_id: round_id + 2
    #                 ],
    #             )
    #             metrics[i]["train_loss"] += train_loss_list
    #             metrics[i]["test_acc"] += test_acc_list
    #         # else:
    #         #     print("Cluster {} has 0 clients in round {}".format(i, round_id))
    #     cluster_map = determine_clustering(ifca_trainers, client_loaders)
    #     if round_id % config["print_freq"] == 0 or round_id == rounds - 1:
    #         print("Curr cluster_map : {}".format(cluster_map))
    #         print("Curr test metrics : cluster 0 : {}, cluster  1 : {}".format(metrics[0]["test_acc"][-1], metrics[1]["test_acc"][-1]))
            
    #     if round_id % config["save_freq"] == 0 or round_id == rounds - 1:

    #         torch.save(
    #             cluster_map,
    #             os.path.join(
    #                 config["results_dir"], "ifca", "cluster_map.pth"
    #             ),
    #         )
    #         torch.save(metrics, os.path.join(config["results_dir"], "ifca", "metrics.pth"))


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
