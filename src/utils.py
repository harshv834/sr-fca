import argparse
import json
import os
import yaml
import numpy as np
import random
import torch
from torch.cuda.amp import autocast
import torch.optim as optim


def args_getter():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="random seed for this experiment"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        # required=True,
        default="synthetic",
        choices=[
            "synthetic",
            "rot_mnist",
            "inv_mnist",
            "rot_cifar10",
            "shakespeare",
            "femnist",
        ],
        help="Dataset to use for this run",
    )
    parser.add_argument(
        "-c",
        "--clustering",
        choices=["sr_fca", "ifca", "cfl", "mocha", "all"],
        # required=True,
        default="sr_fca",
        help="Clustering algo to use for this run, if all is specified run all algorithms and compare",
    )
    parser.add_argument(
        "--dist-metric",
        choices=["euclidean", "cross_entropy"],
        # required=True,
        default="euclidean",
        help="Distance metric to use when comparing models",
    )
    args = parser.parse_args()
    return args
    # ## This is set by hyperparameter config. Do we need a single config file with everything
    # parser.add_argument("-b", "--base-optimizer", choices=["adam", "sgd", "arg"])


def read_algo_config(data_config):
    path = os.path.join(
        "configs",
        "clustering",
        data_config["clustering"],
        data_config["dataset"] + ".yaml",
    )
    assert os.path.exists(path), "Algorithm config does not exist for path {}".format(
        path
    )

    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as err:
            print(err)
    config = {**data_config, **config}
    return config


LOSS_DICT = {"cross_entropy": torch.nn.CrossEntropyLoss(), "mse": torch.nn.MSELoss()}


def compute_metric(model, client_data, train=True, loss=None):
    loader = client_data.trainloader if train else client_data.testloader
    model.eval()
    with torch.no_grad():
        total, total_num = 0.0, 0.0
        for ims, labs in loader:
            with autocast():
                out = model(ims)  # Test-time augmentation
                if loss is not None:
                    total += loss(out, labs)
                else:
                    total += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]

    return total / total_num


def get_optimizer(model_parameters, config):
    optimizer_name = config["optimizer"]["name"]
    optimizer_params = config["optimizer"]["params"]
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model_parameters, **optimizer_params)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model_parameters, **optimizer_params)
    else:
        raise ValueError("Invalid optimizer name {}".format(optimizer_name))
    return optimizer


def get_lr_scheduler(config, optimizer, local_iter, round):
    if config["experiment"] == "rot_cifar10":

        iters_per_epoch = 50000 // int(
            config["batch"]["train"] * (config["num_clients"] // config["num_clusters"])
        )
        epochs = config["init"]["iterations"] // iters_per_epoch
        if round is None:
            lr_schedule = np.interp(
                np.arange((epochs + 1) * iters_per_epoch),
                [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
                [0, 1, 0],
            )
    else:
        lr_schedule = (
            np.ones(config["init"]["iterations"] + 1)
            * config["optimizer"]["params"]["lr"]
        )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    return scheduler


def avg_metrics(metrics_list):
    avg_metric_dict = {}
    for key in metrics_list[0].keys():
        avg_metric_dict[key] = 0.0
    tot_count = 0
    for (count, metric_dict) in metrics_list:
        tot_count += count
        for key in avg_metric_dict.keys():
            avg_metric_dict[key] += count * metric_dict[key]
    for key in avg_metric_dict.keys():
        avg_metric_dict[key] = avg_metric_dict / tot_count
    return avg_metric_dict


def euclidean_dist(w1, w2):
    dist = 0.0
    for key in w1.keys():
        dist += np.linalg.norm(w1[key] - w2[key]) ** 2
    return dist


def compute_dist(trainer_1, trainer_2, client_1, client_2, dist_metric):
    if dist_metric == "euclidean":
        return euclidean_dist(
            trainer_1.model.state_dict(), trainer_2.model.state_dict()
        )
    elif dist_metric == "cross_entropy":
        trainer_1_client_2 = 0.0
        for client in client_2:
            trainer_1_client_2 += trainer_1.compute_loss(client)
        trainer_1_client_2 = trainer_1_client_2 / len(client_2)
        trainer_2_client_1 = 0.0
        for client in client_1:
            trainer_2_client_1 += trainer_2.compute_loss(client)
        trainer_2_client_1 = trainer_2_client_1 / len(client_1)
        return (trainer_1_client_2 + trainer_2_client_1) / 2
    else:
        raise ValueError(
            "{} is not a valid distance metric. Implemented distance metrics are euclidean and cross_entropy".format(
                dist_metric
            )
        )


def correlation_clustering(client_graph, size_threshold):
    clustering = []
    while len(client_graph.nodes) > 0:
        cluster = []
        new_cluster_pivot = random.sample(client_graph.nodes, 1)[0]
        cluster.append(new_cluster_pivot)
        neighbors = client_graph[new_cluster_pivot].copy()
        for node in neighbors:
            cluster.append(node)
            client_graph.remove_node(node)
        client_graph.remove_node(new_cluster_pivot)
        clustering.append(cluster)
    clusters = [cluster for cluster in clustering if len(cluster) > size_threshold]
    cluster_map = {i: clusters[i] for i in range(len(clusters))}
    return cluster_map


def read_data_config(args_dict):
    path = os.path.join("configs", "experiment", args_dict["dataset"] + ".yaml")
    assert os.path.exists(path), "Dataset config does exist for path {}".format(path)

    ## Config loaded from yaml file for the dataset and merged with the arguments from argparse
    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as err:
            print(err)
    config = {**config, **args_dict}
    config["path"] = {
        "data": os.path.join(config["path"], "seed_{}".format(config["seed"]), "data"),
        "results": os.path.join(
            config["path"],
            "seed_{}".format(config["seed"]),
            "results",
            args_dict["clustering"],
        ),
    }
    ## Make directories for data and results
    os.makedirs(config["path"]["data"], exist_ok=True)
    os.makedirs(config["path"]["results"], exist_ok=True)

    return config


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ## What does this do ?
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
