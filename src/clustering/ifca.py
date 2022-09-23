from ..utils import avg_metrics, correlation_clustering, compute_dist
from .base import ClusterFLAlgo
import torch
import os
import networkx as nx
import numpy as np
import itertools
from ..trainers import ClusterTrainer


class IFCA(ClusterFLAlgo):
    def __init__(self, config):
        super(IFCA, self).__init__(config)

    def cluster(self, experiment):
        self.init(experiment)
        for rounds in range(self.config["num_refine_steps"]):
            self.refine(experiment, refine_step)

    def init(self, experiment):
        client_dict = experiment.client_dict
        init_path = os.path.join(self.config["path"]["results"], "init")
        init_metrics = []
        for i in self.client_trainers.keys():
            client_save_dir = os.path.join(init_path, "client_{}".format(i))
            self.client_trainers[i].set_save_dir(client_save_dir)
            client_metrics = self.client_trainers[i].train(
                client_data=client_dict[i],
                local_iter=self.config["init"]["iterations"],
            )
            init_metrics.append((1,client_metrics))
        self.init_metrics = avg_metrics(init_metrics)
        torch.save(self.init_metrics, os.path.join(init_path, "metrics.pth"))

        self.dist_clustering(client_dict, merge=False)
        if len(self.cluster_map.keys()) == 0:
            raise ValueError("Made 0 clusters after INIT")

        torch.save(self.cluster_map, os.path.join(init_path, "cluster_map.pth"))

    def refine(self, experiment, refine_step):
        refine_path = os.path.join(
            self.config["path"]["results"], "refine_{}".format(refine_step)
        )
        client_dict = experiment.client_dict
        self.cluster_trainers = []
        refine_metrics = {}

        for i, client_idx in self.cluster_map.items():
            cluster_trainer = ClusterTrainer(i, self.config)
            cluster_save_dir = os.path.join(refine_path, "cluster_{}".format(i))
            cluster_trainer.set_save_dir(cluster_save_dir)
            cluster_metrics = cluster_trainer.train(
                client_dict=client_dict,
                client_idx=client_idx,
                local_iter = self.config["refine"]["local_iter"],
                rounds = self.config["refine"]["rounds"]
            )
            refine_metrics.append((len(client_idx),cluster_metrics))

        if refine_step == 0:
            self.refine_metrics = {}
        self.refine_metrics[refine_step] = avg_metrics(refine_metrics)

        torch.save(
            self.refine_metrics[refine_step], os.path.join(refine_path, "metrics.pth")
        )

        self.cluster_map = self.recluster()
        if len(self.cluster_map.keys()) == 0:
            raise ValueError("Made 0 clusters after RECLUSTER in Refine step {}".format(refine_step))

        self.dist_clustering(client_dict, merge=True)
        if len(self.cluster_map.keys()) == 0:
            raise ValueError("Made 0 clusters after MERGE in Refine step {}".format(refine_step))
        torch.save(self.cluster_map, os.path.join(refine_path, "cluster_map.pth"))


    def recluster(self, experiment):
        cluster_client_product = {}
        client_dict = experiment.client_dict
        for (i, j) in itertools.product(
            self.cluster_map.keys(), self.client_trainers.keys()
        ):
            cluster_client_product[(i, j)] = compute_dist(
                self.cluster_trainers[i],
                self.client_trainers[j],
                client_dict[self.cluster_map[i]],
                client_dict[j],
                self.config["dist_metric"],
            )
        client_graph = nx.Graph()
        client_graph.add_nodes_from(range(self.config["num_clients"]))
        all_pair_distances = self.compute_pairwise_distances(experiment)
        for (i, j), dist in all_pair_distances.items():
            if dist <= self.config["dist_threshold"]:
                client_graph.add_edge(i, j)
        client_graph = client_graph.to_undirected()
        self.cluster_map = correlation_clustering(
            client_graph, self.config["size_threshold"]
        )
        if len(self.cluster_map.keys()) == 0:
            raise ValueError("Made 0 clusters after INIT")

        cluster_map = {}
        for (i, j), dist in cluster_client_product.items():
            if dist <= self.config["dist_threshold"]:
                if i not in cluster_map.keys():
                    cluster_map[i] = []
                cluster_map[i].append(j)
        j = 0
        self.cluster_map = {}
        for i, cluster in cluster_map.keys():
            if len(cluster) >= self.config["size_threshold"]:
                self.cluster_map[j] = cluster
                j += 1

    def dist_clustering(self,client_dict, merge=False):
        if merge:
            clients = self.cluster_map
            keys = range(self.config["num_clients"])
            trainers = self.cluster_trainers
        else:
            clients = {i: i for i in range(client_dict.keys())}
            keys = self.cluster_map.keys()
            trainers = self.client_trainers
        graph = nx.Graph()
        graph.add_nodes_from(keys)

        for i, j in list(itertools.combinations(keys)):
            dist = compute_dist(trainers[i],trainers[j],
                client_dict[clients[i]],
                client_dict[clients[j]],
                self.config["dist_metric"]
            )
            if dist <= self.config["dist_threshold"]:
                graph.add_edge(i, j)
        graph = graph.to_undirected()
        dist_clustering = correlation_clustering(
            graph, self.config["size_threshold"]
        )
        if merge:
            cluster_map = {}
            for i, cluster in dist_clustering.items():
                new_cluster  = []
                for j in cluster:
                    new_cluster += self.cluster_map[j]
                cluster_map[i] = new_cluster
            self.cluster_map = cluster_map
        else:
            self.cluster_map = dist_clustering
        