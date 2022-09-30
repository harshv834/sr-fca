from src.utils import (
    avg_metrics,
    check_nan,
    wt_dict_dot,
    wt_dict_norm,
    compute_alpha_max,
)
from src.clustering.base import ClusterFLAlgo
import os
import torch
import numpy as np
import itertools
from src.trainers import ClusterTrainer
from time import time
from tqdm import tqdm


class CFL(ClusterFLAlgo):
    def __init__(self, config, tune=False, tune_config=None):
        super(CFL, self).__init__(config, tune, tune_config)
        self.cluster_path = os.path.join(self.config["path"]["results"], "clusters")
        self.cluster_map = {0: range(self.config["num_clients"])}
        self.cluster_trainers = {}

    def cfl_single_node(self, client_dict, cluster_id):
        if len(self.cluster_trainers.keys()) == self.config["num_clusters"]:
            return
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
        if len(self.cluster_map[cluster_id]) == 1:
            return
        else:
            alpha_mat, max_loss_client = self.compute_alpha_mat(cluster_id)
            partitions = self.optimal_bipartitioning(cluster_id, alpha_mat)

            alpha_max_cross = compute_alpha_max(alpha_mat, partitions)
            if (
                max_loss_client >= self.config["client_threshold"]
                and np.sqrt((1 - alpha_max_cross) / 2) > self.config["gamma_max"]
            ) or True:
                _ = self.cluster_map.pop(cluster_id)
                for key, val in partitions.items():
                    self.cluster_map[key] = val
                    self.cfl_single_node(client_dict, key)

    def compute_alpha_mat(self, cluster_id):
        client_idx = self.cluster_map[cluster_id]
        alpha_mat = np.diag(np.ones(len(client_idx)))
        client_wt_diff = self.cluster_trainers[cluster_id].client_wt_diff
        wt_diff_norms = {i: wt_dict_norm(client_wt_diff[i]) for i in client_idx}
        for (i, j) in itertools.combinations(client_idx, 2):
            dot = wt_dict_dot(client_wt_diff[i], client_wt_diff[j])
            alpha_mat[i][j] = dot / (wt_diff_norms[i] * wt_diff_norms[j])
            alpha_mat[j][i] = alpha_mat[i][j]
        max_loss_client = max(wt_diff_norms.values())
        return alpha_mat, max_loss_client

    def optimal_bipartitioning(self, cluster_id, alpha_mat):
        client_idx = self.cluster_map[cluster_id]
        num_clients = len(client_idx)
        alpha_flat = alpha_mat.flatten()
        sorted_idx = (-1 * alpha_flat).argsort()
        C = {i: set([i]) for i in client_idx}
        for i in range(num_clients**2):
            i_1 = client_idx[sorted_idx[i] // num_clients]
            i_2 = client_idx[sorted_idx[i] % num_clients]
            c_temp = set([])
            for j in client_idx:
                if i_1 in C[j] or i_2 in C[j]:
                    c_temp = c_temp.union(C[j])
                    C[j] = set()
            non_empty = 0
            non_empty_id = 0
            for key in C.keys():
                if len(C[key]) > 0:
                    non_empty += 1
                    non_empty_id = key
            if non_empty == 1:
                partition_1_id = cluster_id * 2
                partition_2_id = cluster_id * 2 + 1
                return {
                    partition_1_id: [client_id for client_id in C[non_empty_id]],
                    partition_2_id: [client_id for client_id in c_temp],
                }

    def cluster(self, experiment):
        self.config["time"]["tcluster"] = time()
        client_dict = experiment.client_dict
        init_cluster_id = 0
        self.cluster_metrics = {}
        self.cfl_single_node(client_dict, init_cluster_id)
        self.metrics = []
        for cluster_id in self.cluster_map.keys():
            self.cluster_trainers[cluster_id].client_idx = self.cluster_map[cluster_id]
            metrics = self.cluster_trainers[cluster_id].compute_metrics(client_dict)
            if check_nan(metrics):
                raise ValueError("Nan or inf occurred in metrics")
            self.metrics.append((len(self.cluster_map[cluster_id]), metrics))
        self.metrics = avg_metrics(self.metrics)
        return self.metrics
