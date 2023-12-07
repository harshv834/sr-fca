import itertools
import os
from math import ceil
from time import time

import networkx as nx
import numpy as np
import torch

from src.clustering.base import TRIAL_MAP, ClusterFLAlgo
from src.trainers import ClientTrainer, ClusterTrainer
from src.utils import avg_metrics, check_nan, compute_dist, correlation_clustering
from tqdm import tqdm


class SRFCA(ClusterFLAlgo):
    def __init__(self, config, tune=False, tune_config=None):
        super(SRFCA, self).__init__(config, tune, tune_config)
        if tune:
            self.config["refine"]["rounds"] = ceil(
                int(self.config["init"]["iterations"])
                // int(self.config["refine"]["local_iter"])
            )
        self.client_trainers = {
            i: ClientTrainer(config=self.config, client_id=i, mode="solo")
            for i in range(self.config["num_clients"])
        }

    def cluster(self, experiment):

        self.config["time"]["tcluster"] = time()
        self.init(experiment)
        self.config["time"]["tnew"] = time()
        print(
            "Time taken by INIT step : {} s".format(
                self.config["time"]["tnew"] - self.config["time"]["tcluster"]
            )
        )
        self.config["time"]["tcluster"] = self.config["time"]["tnew"]

        if check_nan(self.init_metrics):
            raise ValueError("Nan or inf occurred in metrics")
            # return self.init_metrics
            # print("Nan or inf occurred in metrics")

        for refine_step in range(self.config["num_refine_steps"]):
            self.refine(experiment, refine_step)
            if check_nan(self.refine_metrics[refine_step]):
                raise ValueError("Nan or inf occurred in metrics")
                # return self.refine_metrics[refine_step]
            self.config["time"]["tnew"] = time()
            print(
                "Time taken by REFINE step{} : {} s".format(
                    refine_step + 1,
                    self.config["time"]["tnew"] - self.config["time"]["tcluster"],
                )
            )
            self.config["time"]["tcluster"] = self.config["time"]["tnew"]

        return self.refine_metrics[self.config["num_refine_steps"] - 1]

    def init(self, experiment):

        client_dict = experiment.client_dict
        init_path = os.path.join(self.config["path"]["results"], "init")

        ## If saved models present in init then start SR_FCA from there
        if "from_init" in self.config.keys() and self.config["from_init"]:
            ## Get the path from init
            for i in tqdm(self.client_trainers.keys()):
                ## Set save directory for local models
                self.client_trainers[i].set_save_dir(os.path.join(init_path, "client_{}".format(i)))
                ## Load saved model weights
                self.client_trainers[i].load_saved_weights()
            self.init_metrics = torch.load(os.path.join(init_path, "metrics.pth"))
        else:
            init_metrics = []
            for i in tqdm(self.client_trainers.keys()):
                client_save_dir = os.path.join(init_path, "client_{}".format(i))
                self.client_trainers[i].set_save_dir(client_save_dir)
                client_metrics = self.client_trainers[i].train(
                    client_data=client_dict[i],
                    local_iter=self.config["init"]["iterations"],
                )
                init_metrics.append((1, client_metrics))
            self.init_metrics = avg_metrics(init_metrics)
            torch.save(self.init_metrics, os.path.join(init_path, "metrics.pth"))

        self.dist_clustering(client_dict, merge=False)
        if len(self.cluster_map.keys()) == 0:
            raise ValueError("Made 0 clusters after INIT")
            # self.cluster_map = TRIAL_MAP

        torch.save(self.cluster_map, os.path.join(init_path, "cluster_map.pth"))

    def refine(self, experiment, refine_step):
        refine_path = os.path.join(
            self.config["path"]["results"], "refine_{}".format(refine_step)
        )
        client_dict = experiment.client_dict
        self.cluster_trainers = {}
        refine_metrics = []

        for i, client_idx in self.cluster_map.items():
            cluster_trainer = ClusterTrainer(self.config, i)
            cluster_save_dir = os.path.join(refine_path, "cluster_{}".format(i))
            cluster_trainer.set_save_dir(cluster_save_dir)
            cluster_metrics = cluster_trainer.train(
                client_dict=client_dict,
                client_idx=client_idx,
                local_iter=self.config["refine"]["local_iter"],
                rounds=self.config["refine"]["rounds"],
            )
            self.cluster_trainers[i] = cluster_trainer
            refine_metrics.append((len(client_idx), cluster_metrics))

        if refine_step == 0:
            self.refine_metrics = {}
        self.refine_metrics[refine_step] = avg_metrics(refine_metrics)

        torch.save(
            self.refine_metrics[refine_step], os.path.join(refine_path, "metrics.pth")
        )

        self.recluster(experiment)
        self.empty_cluster_check("REFINE step {}".format(refine_step))
        if len(self.cluster_map.keys()) == 0:
            raise ValueError(
                "Made 0 clusters after RECLUSTER in Refine step {}".format(refine_step)
            )
            # self.cluster_map = TRIAL_MAP

        self.dist_clustering(client_dict, merge=True)
        import ipdb;ipdb.set_trace()
        if len(self.cluster_map.keys()) == 0:
            raise ValueError(
                "Made 0 clusters after MERGE in Refine step {}".format(refine_step)
            )
            # self.cluster_map = TRIAL_MAP
        torch.save(self.cluster_map, os.path.join(refine_path, "cluster_map.pth"))

    def recluster(self, experiment):
        cluster_client_product = {}
        client_dict = experiment.client_dict
        for (cluster_id, client_id) in itertools.product(
            self.cluster_map.keys(), self.client_trainers.keys()
        ):

            dist = compute_dist(
                self.cluster_trainers[cluster_id],
                self.client_trainers[client_id],
                [client_dict[key] for key in self.cluster_map[cluster_id]],
                [client_dict[client_id]],
                self.config["dist_metric"],
            )
            if client_id not in cluster_client_product.keys():
                cluster_client_product[client_id] = {cluster_id: dist}
            else:
                cluster_client_product[client_id][cluster_id] = dist
        clusters = {}
        for client_id in self.client_trainers.keys():
            sorted_cluster_idx = sorted(cluster_client_product[client_id].keys())
            cluster_dists = np.array(
                [
                    cluster_client_product[client_id][cluster_id]
                    for cluster_id in sorted_cluster_idx
                ]
            )
            cluster_id = sorted_cluster_idx[cluster_dists.argmin()]
            clusters[client_id] = cluster_id

        cluster_map = {cluster_id: [] for cluster_id in self.cluster_map.keys()}
        for client_id, cluster_id in clusters.items():
            cluster_map[cluster_id].append(client_id)

        j = 0
        self.cluster_map = {}
        for cluster in cluster_map.values():
            if len(cluster) >= self.config["size_threshold"]:
                self.cluster_map[j] = cluster
                j += 1

    def dist_clustering(self, client_dict, merge=False):
        if merge:
            clients = self.cluster_map
            trainers = self.cluster_trainers
        else:
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
        # print("Min dist",sorted(dist_dict.values())[0])
        # print("Max dist",sorted(dist_dict.values())[-1])
        if not merge:
            if "dist_threshold" not in self.config.keys():
                self.config["dist_threshold"] = sorted(dist_dict.values())[
                    ceil(self.config["dist_fraction"] * len(dist_dict.keys()))
                ]
        for (i, j), dist in dist_dict.items():
            if dist <= self.config["dist_threshold"]:
                graph.add_edge(i, j)
        graph = graph.to_undirected()
        dist_clustering = correlation_clustering(
            graph, 0 if merge else self.config["size_threshold"]
        )
        if merge:
            import ipdb;ipdb.set_trace()
            cluster_map = {}
            for i, cluster in dist_clustering.items():
                new_cluster = []
                for j in cluster:
                    new_cluster += self.cluster_map[j]
                cluster_map[i] = new_cluster
            self.cluster_map = cluster_map
        else:
            self.cluster_map = dist_clustering
