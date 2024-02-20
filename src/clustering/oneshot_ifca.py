import itertools
import os
import shutil
from math import ceil
from time import time

import numpy as np
import networkx as nx
import torch
from tqdm import tqdm

from src.clustering.base import ClusterFLAlgo
from src.trainers import ClusterTrainer, ClientTrainer
from src.utils import avg_metrics, check_nan, correlation_clustering,compute_dist



class OneShotIFCA(ClusterFLAlgo):
    def __init__(self, config, tune=False, tune_config=None):
        super(OneShotIFCA, self).__init__(config, tune, tune_config)
        if tune:
            self.config["rounds"] = ceil(
                self.config["iterations"] / self.config["local_iter"]
            )


    def cluster(self, experiment):

        self.config["time"]["tcluster"] = time()
        client_dict = experiment.client_dict
        
        self.init_cluster_map(client_dict)
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

    def init_cluster_map(self, client_dict):
        ## Create local client trainers to load model and perform dist clustering
        local_path = os.path.join(self.config["path"]["results"], "init")
        client_path = lambda i, path : os.path.join(path, "client_{}".format(i))
        ## Get the path from init
        init_path = "/" + os.path.join(*(self.config["path"]["results"].split("/")[:-1] + ["sr_fca", "init"]))
        self.client_trainers = {
            i: ClientTrainer(config=self.config, client_id=i, mode="solo")
            for i in range(self.config["num_clients"])
        }
        for i in tqdm(self.client_trainers.keys()):
            ## Set save directory for local models
            self.client_trainers[i].set_save_dir(client_path(i, local_path))
            model_path = os.path.join(client_path(i, init_path), "model.pth")

            ## Copy metrics and model of local model
            shutil.copy2(model_path, os.path.join(client_path(i, local_path), "model.pth"))
            
            ## Load saved model weights
            self.client_trainers[i].load_saved_weights()
        ## dist clustering
        self.dist_clustering(client_dict)
        if len(self.cluster_map.keys()) == 0:
            raise ValueError("Made 0 clusters after INIT")
            # self.cluster_map = TRIAL_MAP
        ## Set number of clusters
        self.config["num_clusters"] = len(self.cluster_map)

    def dist_clustering(self, client_dict):
        clients = {i: [i] for i in client_dict.keys()}
        trainers = self.client_trainers
        keys = list(clients.keys())
        graph = nx.Graph()
        graph.add_nodes_from(keys)
        dist_dict = {}
        for i, j in tqdm(list(itertools.combinations(keys, 2))):
            dist = compute_dist(
                trainers[i],
                trainers[j],
                [client_dict[key] for key in clients[i]],
                [client_dict[key] for key in clients[j]],
                self.config["dist_metric"],
            )
            dist_dict[(i, j)] = dist
        if "dist_threshold" not in self.config.keys():
            self.config["dist_threshold"] = sorted(dist_dict.values())[
                ceil(self.config["dist_fraction"] * len(dist_dict.keys()))
            ]
        for (i, j), dist in dist_dict.items():
            if dist <= self.config["dist_threshold"]:
                graph.add_edge(i, j)
        graph = graph.to_undirected()
        dist_clustering = correlation_clustering(graph, self.config["size_threshold"])
        self.cluster_map = dist_clustering

