import itertools
import os
from math import ceil
from time import time

import numpy as np
import torch
from tqdm import tqdm

from src.clustering.base import ClusterFLAlgo
from src.trainers import ClusterTrainer
from src.utils import avg_metrics, check_nan


class IFCA(ClusterFLAlgo):
    def __init__(self, config, tune=False, tune_config=None):
        super(IFCA, self).__init__(config, tune, tune_config)
        if tune:
            self.config["rounds"] = ceil(
                self.config["iterations"] / self.config["local_iter"]
            )
        self.init_cluster_map()
        self.cluster_trainers = {}
        self.cluster_path = os.path.join(self.config["path"]["results"], "clusters")

        for cluster_id in range(self.config["num_clusters"]):
            cluster_trainer = ClusterTrainer(self.config, cluster_id)
            cluster_save_dir = os.path.join(
                self.cluster_path, "cluster_{}".format(cluster_id)
            )
            cluster_trainer.set_save_dir(cluster_save_dir)
            self.cluster_trainers[cluster_id] = cluster_trainer

    def cluster(self, experiment):
        self.config["time"]["tcluster"] = time()
        client_dict = experiment.client_dict
        self.round_metrics = {}
        for round_id in tqdm(range(self.config["rounds"])):
            if round_id != 0:
                self.min_loss_clustering(client_dict)
                self.empty_cluster_check("round id : {}".format(round_id))
            torch.save(
                self.cluster_map,
                os.path.join(self.config["path"]["results"], "cluster_map.pth"),
            )

            self.round_metrics[round_id] = []
            for cluster_id in range(len(self.cluster_trainers)):
                if len(self.cluster_map[cluster_id]) > 0:
                    metrics = self.cluster_trainers[cluster_id].train(
                        client_dict=client_dict,
                        client_idx=self.cluster_map[cluster_id],
                        local_iter=self.config["local_iter"],
                        rounds=(round_id, round_id + 1),
                    )
                    if check_nan(metrics):
                        # return metrics
                        raise ValueError("Nan or inf occurred in metrics")
                    self.round_metrics[round_id].append(
                        (len(self.cluster_map[cluster_id]), metrics)
                    )
            self.round_metrics[round_id] = avg_metrics(self.round_metrics[round_id])
            if (
                round_id % self.config["freq"]["save"] == 0
                or round_id == self.config["rounds"] - 1
            ):
                torch.save(
                    self.round_metrics,
                    os.path.join(self.config["path"]["results"], "metrics.pth"),
                )

        return self.round_metrics[self.config["rounds"] - 1]

    def min_loss_clustering(self, client_dict):
        client_cluster_metrics = {}
        for client_id, cluster_id in itertools.product(
            range(self.config["num_clients"]), self.cluster_map.keys()
        ):
            if client_id not in client_cluster_metrics.keys():
                client_cluster_metrics[client_id] = np.inf * np.ones(
                    self.config["num_clusters"]
                )
            client_cluster_metrics[client_id][cluster_id] = self.cluster_trainers[
                cluster_id
            ].compute_loss(client_dict[client_id], train=True)
        for cluster_id in self.cluster_map.keys():
            self.cluster_map[cluster_id] = []
        for client_id in range(self.config["num_clients"]):
            cluster_id = client_cluster_metrics[client_id].argmin()
            self.cluster_map[cluster_id].append(client_id)

    def init_cluster_map(self):
        self.cluster_map = {
            cluster_id: [] for cluster_id in range(self.config["num_clusters"])
        }
        client_idx = np.arange(self.config["num_clients"])
        np.random.shuffle(client_idx)
        for client_id in client_idx:
            self.cluster_map[client_id % self.config["num_clusters"]].append(client_id)
