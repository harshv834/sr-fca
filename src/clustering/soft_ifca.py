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


class SoftIFCA(ClusterFLAlgo):
    def __init__(self, config, tune=False, tune_config=None):
        super(SoftIFCA, self).__init__(config, tune, tune_config)
        if tune:
            self.config["rounds"] = ceil(
                self.config["iterations"] / self.config["local_iter"]
            )
        self.cluster_trainers = {}
        self.config["path"]["results"] = os.path.join(self.config["path"]["results"], str(self.config["num_clusters"]))
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
        
        ## initialize agg and clustering weights.
        self.init_cluster_wts(client_dict)

        self.round_metrics = {}
        for round_id in tqdm(range(self.config["rounds"])):
            
            if round_id != 0:
                self.soft_min_loss_clustering(client_dict)
            #     self.empty_cluster_check("round id : {}".format(round_id))
            # torch.save(
            #     self.cluster_wts,
            #     os.path.join(self.config["path"]["results"], "cluster_wts.pth"),
            # )
            ## Sampling weights assigned in proportion with number of samples per client.
            self.sampling_wts = self.cluster_wts * self.client_num_samples
            self.sampling_wts = self.sampling_wts/self.sampling_wts.sum(axis=1).reshape(-1,1)

            self.round_metrics[round_id] = []
            for cluster_id in range(len(self.cluster_trainers)):
                ## what is the equivalent of an empty cluster in soft clustering setup.
                #if len(self.cluster_map[cluster_id]) > 0:
                    ## Change trainer to sample clients with client_sampling_wts
                metrics = self.cluster_trainers[cluster_id].train(
                    client_dict=client_dict,
                    client_idx=list(client_dict.keys()),
                    local_iter=self.config["local_iter"],
                    rounds=(round_id, round_id + 1),
                    client_sampling_wts = self.sampling_wts[cluster_id]
                )
                if check_nan(metrics):
                    # return metrics
                    raise ValueError("Nan or inf occurred in metrics")
                ## Round metrics averaged with average weight of the cluster as all clusters used for all clients at all times. Not sure if this is the best method.
                self.round_metrics[round_id].append(
                    (self.cluster_wts[cluster_id].sum(), metrics)
                )
            ## Added a total count to metric averaging to prevent total count being 0
            self.round_metrics[round_id] = avg_metrics(self.round_metrics[round_id],min_tot_count=self.config["sigma"])
            if (
                round_id % self.config["freq"]["save"] == 0
                or round_id == self.config["rounds"] - 1
            ):
                torch.save(
                    self.round_metrics,
                    os.path.join(self.config["path"]["results"], "metrics.pth"),
                )
        return self.round_metrics[self.config["rounds"] - 1]

    ## Changed this to soft loss clustering per datapoint.
    def soft_min_loss_clustering(self, client_dict):
        client_cluster_metrics = {client_id : np.inf*np.ones((self.config["num_clusters"],self.client_num_samples[client_id])) for client_id in client_dict.keys()}

        for cluster_id in range(self.config["num_clusters"]):
            ## Compute list of losses of all clients on given cluster model.
            cluster_loss_list = self.cluster_trainers[cluster_id].compute_loss_list(client_dict, train=True)
            for client_id in client_dict.keys():
                ## Assign each client loss to correct index in client_cluster_metrics.
                client_cluster_metrics[client_id][cluster_id] = cluster_loss_list[client_id]
        
        for client_id in client_dict.keys():
            ## Cluster weights for every client updated based on 
            cluster_counts  = np.argmin(client_cluster_metrics[client_id], axis=1)
            self.cluster_wts[:, client_id] = np.maximum(cluster_counts/cluster_counts.sum(), self.config["sigma"])    
            

    def init_cluster_wts(self, client_dict):
        self.client_num_samples = np.array([client.train_size for client in client_dict.values()])
        
        ## Importance weights assigned by each client to a given cluster, sums to 1 for every client
        self.cluster_wts = np.ones((self.config["num_clusters"], self.config["num_clients"]))/self.config["num_clusters"]
