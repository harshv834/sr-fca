import itertools
import os
from math import ceil
from time import time

import numpy as np
import torch
from tqdm import tqdm

from src.clustering.base import ClusterFLAlgo
from src.trainers import ClusterTrainer
from src.utils import avg_metrics, check_nan, wt_dict_mean


class FedDrift(ClusterFLAlgo):
    def __init__(self, config, tune=False, tune_config=None):
        super(FedDrift, self).__init__(config, tune, tune_config)
        if tune:
            self.config["rounds"] = ceil(
                self.config["iterations"] / self.config["local_iter"]
            )
        self.init_cluster_map()
        self.cluster_trainers = {}
        self.config["path"]["results"] = os.path.join(self.config["path"]["results"], str(self.config["num_clusters"]))
        self.cluster_path = os.path.join(self.config["path"]["results"], "clusters")
        self.new_cluster_clients = []
        self.min_client_loss = np.inf* np.ones(self.config["num_clients"])
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
            self.min_loss_clustering(client_dict, round_id)
            self.add_new_cluster(round_id)
            self.empty_cluster_check("round id : {}".format(round_id))
            torch.save(
                self.cluster_map,
                os.path.join(self.config["path"]["results"], "cluster_map.pth"),
            )
            print("num of clusters", len(self.cluster_trainers))
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

    def min_loss_clustering(self, client_dict, round_id):
        self.new_cluster_clients = []
        client_cluster_metrics = {} 
        sorted_cluster_idx = sorted(list(self.cluster_map.keys()))
        cluster_losses = {cluster_id : {diff_cluster_id : 0.0 for diff_cluster_id in self.cluster_map.keys()} for cluster_id in self.cluster_map.keys()} 
        
        self.cluster_distances = {(i,j) : 0.0 for i,j in itertools.combinations(sorted_cluster_idx,2)}
        for client_id, cluster_id in itertools.product(
            range(self.config["num_clients"]), sorted(list(self.cluster_map.keys()))
        ):
            if client_id not in client_cluster_metrics.keys():
                client_cluster_metrics[client_id] = np.inf * np.ones(
                    self.config["num_clusters"]
                )
            client_cluster_metrics[client_id][cluster_id] = self.cluster_trainers[
                cluster_id
            ].compute_loss(client_dict[client_id], train=True)
        ## Don't do clustering first round.
        if round_id != 0:
            for cluster_id in self.cluster_map.keys():
                self.cluster_map[cluster_id] = []

        for client_id in range(self.config["num_clients"]):
            argmin_idx = client_cluster_metrics[client_id].argmin()
            cluster_id = sorted_cluster_idx[argmin_idx]
            client_cluster_loss = client_cluster_metrics[client_id][argmin_idx]
            if round_id != 0:
                ## Delta shit worked out.
                if client_cluster_loss - self.min_client_loss[client_id] > self.config["delta"]:
                    self.new_cluster_clients.append(client_id)
                else:
                    self.cluster_map[cluster_id].append(client_id)
                    ## Record previous loss.            
                    cluster_losses[cluster_id][cluster_id] += client_cluster_metrics[client_id][cluster_id]
                    ## Compute sum of pairwise losses.
                    for diff_id in range(len(self.cluster_map.keys())):
                        diff_cluster_id = sorted_cluster_idx[diff_id]
                        if diff_cluster_id != cluster_id:
                            cluster_losses[diff_cluster_id][cluster_id] += client_cluster_metrics[client_id][diff_id]
            self.min_client_loss[client_id] = client_cluster_metrics[client_id][cluster_id]
        if round_id == 0:
            return
        ## Compute average of pairwise losses.
        for cluster_id in self.cluster_map.keys():
            num_clients_in_cluster = len(self.cluster_map[cluster_id])
            if num_clients_in_cluster > 0:
                for diff_id in range(self.config["num_clusters"]):
                    diff_cluster_id = sorted_cluster_idx[diff_id]
                    cluster_losses[diff_cluster_id][cluster_id]/num_clients_in_cluster

        ## Compute cluster distances and merge clusters.
        for i, j in itertools.combinations(sorted_cluster_idx,2):
            if len(self.cluster_map[i]) >0 and len(self.cluster_map[j]) > 0:
                self.cluster_distances[(i,j)]  = max(cluster_losses[i][j] - cluster_losses[j][i], cluster_losses[j][i] - cluster_losses[i][j],0)
        
        for i,j in itertools.combinations(range(len(sorted_cluster_idx)),2):
            if len(self.cluster_map[i]) >0 and len(self.cluster_map[j]) > 0:
                if self.cluster_distances[(i,j)] <= self.config["delta"]:
                    self.merge_clusters(i,j)


    ## Add new cluster with single client
    def add_new_cluster(self, round_id):
        if round_id == 0:
            return
        for client_id in self.new_cluster_clients:
            import ipdb;ipdb.set_trace()
            max_cluster_id = max(list(self.cluster_map.keys()))
            new_cluster_id = max_cluster_id +1
            self.cluster_map[new_cluster_id] = [client_id]
            new_cluster_trainer = ClusterTrainer(self.config, new_cluster_id)
            cluster_save_dir = os.path.join(
                self.cluster_path, "cluster_{}".format(new_cluster_id)
            )
            new_cluster_trainer.set_save_dir(cluster_save_dir)
            self.cluster_trainers[new_cluster_id] = new_cluster_trainer

    ## Merge clusters with same client.
    def merge_clusters(self, i, j):
        ## Itertools gives largest number at end (j is largest)
        import ipdb;ipdb.set_trace()
        new_cluster_id = j
        self.cluster_map[new_cluster_id] = self.cluster_map[i] + self.cluster_map[j]
        del self.cluster_map[i]
        avg_model_wt = wt_dict_mean(self.cluster_trainers[i].get_model_wts(), self.cluster_trainers[j].get_model_wts())
        self.cluster_trainers[j].load_wts_from_dict(avg_model_wt)
        del self.cluster_trainers[i]
        for cluster_id in self.cluster_map.keys():
            i_pair = (i,cluster_id) if cluster_id > i else (cluster_id, i)
            j_pair = (j, cluster_id) if cluster_id > j else (cluster_id, j)
            self.cluster_distances[j_pair] = max(self.cluster_distances[i_pair], self.cluster_distances[j_pair])
            del self.cluster_distances[i_pair]

        
    def init_cluster_map(self):
        self.cluster_map = {
            cluster_id: [] for cluster_id in range(self.config["num_clusters"])
        }
        client_idx = np.arange(self.config["num_clients"])
        np.random.shuffle(client_idx)
        for client_id in client_idx:
            self.cluster_map[client_id % self.config["num_clusters"]].append(client_id)
