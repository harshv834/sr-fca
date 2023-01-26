import itertools
import os
from math import ceil
from time import time

import numpy as np


from src.clustering.base import ClusterFLAlgo
from src.trainers import ClusterTrainer
from src.utils import (
    avg_metrics,
    check_nan,
    compute_alpha_max,
    wt_dict_dot,
    wt_dict_norm,
)
import torch


class CFL(ClusterFLAlgo):
    def __init__(self, config, tune=False, tune_config=None):
        super(CFL, self).__init__(config, tune, tune_config)
        if tune:
            self.config["rounds"] = ceil(
                self.config["iterations"] / self.config["local_iter"]
            )
        self.cluster_path = os.path.join(self.config["path"]["results"], "clusters")
        self.cluster_map = {}
        self.cluster_trainers = {}
        self.cluster_metrics = {}
        self.cluster_idx_to_train = []

    def cfl_single_node(self, client_dict, cluster_id):

        ## Train a model for the cluster
        cluster_trainer = ClusterTrainer(
            self.config, cluster_id, stop_threshold=self.config["stop_threshold"]
        )
        cluster_save_dir = os.path.join(
            self.cluster_path, "cluster_{}".format(cluster_id)
        )
        cluster_trainer.set_save_dir(cluster_save_dir)
        self.cluster_metrics[cluster_id] = cluster_trainer.train(
            client_dict=client_dict,
            client_idx=self.cluster_map[cluster_id],
            local_iter=self.config["local_iter"],
            rounds=self.config["rounds"],
        )
        self.cluster_trainers[cluster_id] = cluster_trainer

        ## Split cluster into two parts
        if (
            len(self.cluster_map[cluster_id]) == 1
            or len(self.cluster_map) == self.config["num_clusters"]
        ):
            return
        else:
            ## Compute alpha for every client pair in cluster
            alpha_mat, max_loss_client = self.compute_alpha_mat(cluster_id)
            if np.isnan(alpha_mat).any() or np.isinf(alpha_mat).any():
                print(
                    "Nan or inf occurred in alpha for cluster : {}".format(cluster_id)
                )
            ## Obtain optimal bipartitioning to maximize
            partitions = self.optimal_bipartitioning(cluster_id, alpha_mat)

            ## Obtain max alpha between two partitions
            alpha_max_cross = compute_alpha_max(
                alpha_mat, partitions, self.cluster_map[cluster_id]
            )
            if (
                max_loss_client >= self.config["client_threshold"]
                and np.sqrt((1 - alpha_max_cross) / 2) > self.config["gamma_max"]
            ) or True:
                _ = self.cluster_map.pop(cluster_id)
                _ = self.cluster_trainers.pop(cluster_id)
                for key, val in partitions.items():
                    self.cluster_map[key] = val
                    self.cluster_idx_to_train.append(key)
            return

    def compute_alpha_mat(self, cluster_id):
        """Compute alpha matrix which is cosine similarity of loss gradient of different clients at optima for the cluster.

        Args:
            cluster_id (_type_): _description_

        Returns:
            _type_: final alpha matric
        """

        ## Cosine similarity is 1 if the two clients are same, so start
        ## with an identity matrix
        client_idx = self.cluster_map[cluster_id]
        alpha_mat = np.diag(np.ones(len(client_idx)))

        ## Compute weight differences/loss gradient at optima for clients in given cluster
        client_wt_diff = self.cluster_trainers[cluster_id].client_wt_diff
        wt_diff_norms = {i: wt_dict_norm(client_wt_diff[i]) for i in client_idx}
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

    def optimal_bipartitioning(self, cluster_id, alpha_mat):
        client_idx = self.cluster_map[cluster_id]
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

    def cluster(self, experiment):
        """Main method to create clusters of clients

        Args:
            experiment (dict): Dict of client data used for the experiment

        Raises:
            ValueError: When Nan or inf appears in metrics

        Returns:
            dict: Metrics of trained cluster federated learning 
        """
        ### Initialize the client dict and put all clients inside the first cluster which has cluster_id 0
        self.config["time"]["tcluster"] = time()

        client_dict = experiment.client_dict
        init_cluster_id = 0
        self.cluster_map = {init_cluster_id: list(range(self.config["num_clients"]))}

        ## Add this cluster_id to a FIFO queue which contains cluster_idx to train next
        self.cluster_idx_to_train.append(init_cluster_id)

        ## While required number of clusters haven't been trained, perform CFL on a new cluster id
        while len(self.cluster_trainers.keys()) < self.config["num_clusters"]:
            ## Put cluster_idx to train in a queue and pop the queue and train each cluster.
            if len(self.cluster_idx_to_train) > 0:
                cluster_idx_to_train = self.cluster_idx_to_train.pop(0)
                client_dict_to_train = {
                    client_idx: client_dict[client_idx]
                    for client_idx in self.cluster_map[cluster_idx_to_train]
                }
                self.cfl_single_node(client_dict_to_train, cluster_idx_to_train)
            else:
                break
        ## Among the final clusters which remain,
        self.metrics = []
        for cluster_id in self.cluster_map.keys():
            self.cluster_trainers[cluster_id].client_idx = self.cluster_map[cluster_id]
            metrics = self.cluster_trainers[cluster_id].compute_metrics(client_dict)
            if check_nan(metrics):
                raise ValueError("Nan or inf occurred in metrics")
            self.metrics.append((len(self.cluster_map[cluster_id]), metrics))
        self.metrics = avg_metrics(self.metrics)
        torch.save(self.metrics, os.path.join(self.config["path"]["results"], "metrics.pth"))
        torch.save(self.cluster_map, os.path.join(self.config["path"]["results"], "cluster_map.pth"))
        return self.metrics
