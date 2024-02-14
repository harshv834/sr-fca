import torch
import numpy as np
from tqdm import tqdm
import os
import random
import argparse
import networkx as nx
import itertools

from cifar_dataset import  create_client_loaders
from cifar_utils import  cross_entropy_metric, create_config, model_weights_diff
from cifar_trainers import ClientTrainer, ClusterTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--het", choices=["rot", "label"], type=str, required=True, 
                    help="Choose whether to run experiments on Rotated CIFAR10 or CIFAR10 with label heterogeneity"
)
parser.add_argument(
    "--seed", type=int, required=True, help="Random seed for the experiment"
)
parser.add_argument(
    "--from_init",
    action=argparse.BooleanOptionalAction  
)

clustering = []

def correlation_clustering(G):
    global clustering
    if len(G.nodes) == 0:
        return
    else:
        cluster = []
        new_cluster_pivot = random.sample(G.nodes, 1)[0]
        cluster.append(new_cluster_pivot)
        neighbors = G[new_cluster_pivot].copy()
        for node in neighbors:
            cluster.append(node)
            G.remove_node(node)
        G.remove_node(new_cluster_pivot)
        clustering.append(cluster)
        correlation_clustering(G)



config= create_config(parser)
### Define beta here for cluster trainer
config["beta"] = 0

client_loaders = create_client_loaders(config)

# model = model.to(memory_format=torch.channels_last).cuda()

init_path = os.path.join(config["results_dir"], "init")
client_trainers = [
    ClientTrainer(
        config, os.path.join(init_path, "node_{}".format(i)), i
    )
    for i in range(config["num_clients"])
]

## If saved models present in init then start SR_FCA from there
if args.from_init:
    ## Get the path from init
    for i in tqdm(range(config["num_clients"])):
        ## Load saved model weights
        client_trainers[i].load_saved_weights()
    # self.init_metrics = torch.load(os.path.join(init_path, "metrics.pth"))
else:
    for i in tqdm(range(config["num_clients"])):
        client_trainers[i].train(client_loaders[i])

G = nx.Graph()
G.add_nodes_from(range(config["num_clients"]))

wt = client_trainers[0].model.state_dict()
# thresh = 0
# for key in wt.keys():
#     thresh += wt[key].norm()**2
# print(torch.sqrt(thresh))
# thresh = 37.68

all_pairs = list(itertools.combinations(range(config["num_clients"]), 2))
arr = {}
for pair in tqdm(all_pairs):
    arr[pair] = cross_entropy_metric(
        client_trainers[pair[0]],
        client_trainers[pair[1]],
        [client_loaders[pair[0]]],
        [client_loaders[pair[1]]],
    )

    import ipdb;ipdb.set_trace()
    w_1  = client_trainers[pair[0]].model.state_dict()
    w_2 = client_trainers[pair[1]].model.state_dict()
    norm_diff = model_weights_diff(w_1, w_2)
    arr.append(norm_diff)
    
#thresh = torch.mean(torch.tensor(arr))
thresh = arr[torch.tensor(arr).argsort()[int(0.3*len(arr))-1]]
thresh = sorted(all_pairs.values())[int(0.3 * len(all_pairs))]
thresh = 0.0186

for pair in all_pairs:
    if arr[pair] < thresh:
        G.add_edge(pair[0], pair[1])
G = G.to_undirected()
# cliques = list(nx.algorithms.clique.enumerate_all_cliques(G))

correlation_clustering(G)

# config["t"] = 7
# t = config["t"]
clusters = [cluster for cluster in clustering if len(cluster) > 1]
cluster_map = {i: clusters[i] for i in range(len(clusters))}
# import ipdb;ipdb.set_trace()
config["refine_steps"] = 1
for refine_step in tqdm(range(config["refine_steps"])):
    cluster_trainers = []
    refine_path = os.path.join(config["results_dir"], "refine_{}".format(refine_step))
    os.makedirs(refine_path, exist_ok=True)
    for cluster_id in tqdm(cluster_map.keys()):
        cluster_clients = [client_loaders[i] for i in cluster_map[cluster_id]]
        cluster_trainer = ClusterTrainer(
            config,
            os.path.join(
                refine_path,
                "cluster_{}".format(cluster_id),
            ),
            cluster_id,
        )
        cluster_trainer.train(cluster_clients)
        cluster_trainers.append(cluster_trainer)
    torch.save(cluster_map, os.path.join(refine_path, "cluster_maps.pth"))
    # import ipdb

    # ipdb.set_trace()
    cluster_map_recluster = {}
    for key in cluster_map.keys():
        cluster_map_recluster[key] = []

    for i in tqdm(range(config["num_clients"])):
        trainer_node = client_trainers[i]

        dist_diff = np.infty
        new_cluster_id = 0
        for cluster_id in cluster_map.keys():
            trainer_cluster = cluster_trainers[cluster_id]
            cluster_clients = [client_loaders[i] for i in cluster_map[cluster_id]]
            curr_dist_diff = cross_entropy_metric(
                trainer_node, trainer_cluster, [client_loaders[i]], cluster_clients
            )
            if dist_diff > curr_dist_diff:
                new_cluster_id = cluster_id
                dist_diff = curr_dist_diff

        cluster_map_recluster[new_cluster_id].append(i)
    keys = list(cluster_map_recluster.keys()).copy()
    for key in keys:
        if len(cluster_map_recluster[key]) == 0:
            cluster_map_recluster.pop(key)
    cluster_map = cluster_map_recluster

    G = nx.Graph()
    G.add_nodes_from(cluster_map.keys())

    all_pairs = list(itertools.combinations(cluster_map.keys(), 2))
    arr = {}
    for pair in tqdm(all_pairs):

        cluster_1 = [client_loaders[i] for i in cluster_map[pair[0]]]
        cluster_2 = [client_loaders[i] for i in cluster_map[pair[1]]]

        arr[pair] = cross_entropy_metric(
            cluster_trainers[pair[0]],
            client_trainers[pair[1]],
            cluster_1,
            cluster_2,
        )

    for pair in all_pairs:
        if arr[pair] < thresh:
            G.add_edge(pair[0], pair[1])
    G = G.to_undirected()

    clustering = []

    correlation_clustering(G)
    merge_clusters = [cluster for cluster in clustering if len(cluster) > 0]

    # merge_cluster_map = {i: clusters[i] for i in range(len(clusters))}
    # clusters = list(nx.algorithms.clique.enumerate_all_cliques(G))
    cluster_map_new = {}
    for i in range(len(merge_clusters)):
        cluster_map_new[i] = []
        for j in merge_clusters[i]:
            cluster_map_new[i] += cluster_map[j]
    cluster_map = cluster_map_new

# global_trainer = GlobalTrainer(config, os.path.join(config["results_dir"], "global"))
# global_trainer.train(client_loaders)
