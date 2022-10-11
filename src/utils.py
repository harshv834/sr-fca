import argparse
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from math import ceil
from shutil import rmtree

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import yaml
from pytorch_lightning.utilities.seed import seed_everything
from torch.cuda.amp import autocast
from ray_lightning import RayStrategy
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)


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
        choices=["sr_fca", "ifca", "cfl", "mocha", "fedavg", "all"],
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
    args = vars(args)
    return args
    # ## This is set by hyperparameter config. Do we need a single config file with everything
    # parser.add_argument("-b", "--base-optimizer", choices=["adam", "sgd", "arg"])


def read_algo_config(data_config, tune=False):
    path = os.path.join(
        "configs",
        "clustering",
        data_config["clustering"],
        data_config["dataset"]["name"],
        "best_hp.yaml",
    )
    print("here")
    exists = os.path.exists(path)
    if tune:
        if exists:
            rmtree(path)

        # config = suggest_config(
        #     data_config["dataset"]["name"], data_config["clustering"]
        # )
    else:
        assert exists, "Algorithm config does not exist for path {}".format(path)

        with open(path, "r") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as err:
                print(err)
    ## Works only on python >=3.9.0
    config = config | data_config

    if config["dataset"]["name"] == "synthetic":
        config["model"]["params"] = {
            "dimension": config["dataset"]["dimension"],
            "scale": config["dataset"]["scale"],
        }
    return config


LOSS_DICT = {"cross_entropy": torch.nn.CrossEntropyLoss(), "mse": torch.nn.MSELoss()}


def compute_metric(
    model, client_data, train=True, loss=None, device=None, lstm_flag=False
):
    loader = client_data.trainloader if train else client_data.testloader
    model.eval()
    if device is not None:
        model = model.to(memory_format=torch.channels_last).to(device)
    if lstm_flag:
        batch_size, hidden = None, None
    with torch.no_grad():
        total, total_num = 0.0, 0.0
        for X, Y in loader:
            if lstm_flag:
                if batch_size is None or X.shape[0] != batch_size:
                    batch_size = X.shape[0]
                    hidden = model.zero_state(batch_size)
            if device is not None:
                X, Y = X.to(device), Y.to(device)
                if lstm_flag:
                    hidden = (hidden[0].to(device), hidden[1].to(device))

            with autocast():
                if lstm_flag:
                    out, hidden = model(X, hidden)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                else:
                    out = model(X)  # Test-time augmentation

                if loss is not None:
                    total += loss(out, Y).detach()
                else:
                    total += out.argmax(1).eq(Y).sum().detach()
                total_num += Y.shape[0]

    model.to("cpu")
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


def get_lr_scheduler(optimizer, config, local_iter, round_id):
    cond_1 = config["dataset"]["name"] == "rot_cifar10"
    cond_2 = config["clustering"] == "sr_fca"
    if cond_1 and cond_2:
        iters_per_epoch = 50000 // int(
            config["batch"]["train"]
            * (config["num_clients"] // config["dataset"]["num_clusters"])
        )
        epochs = config["init"]["iterations"] // iters_per_epoch
        lr_schedule = np.interp(
            np.arange((epochs + 1) * iters_per_epoch),
            [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
            [0, 1, 0],
        )
        if round_id is not None:
            if type(round_id) == tuple:
                first_iter = round_id[0] * local_iter
                last_iter = max((round_id[1] + 1) * local_iter, lr_schedule.shape[0])
            else:
                first_iter = 0
                last_iter = max((round_id + 1) * local_iter, lr_schedule.shape[0])
            lr_schedule = lr_schedule[first_iter:last_iter]
    elif not cond_1 and cond_2:
        lr_schedule = np.ones(config["init"]["iterations"] + 1)
    else:
        lr_schedule = np.ones(max(config["rounds"], local_iter) + 1)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    return scheduler


def avg_metrics(metrics_list):
    avg_metric_dict = {}

    metric_keys = metrics_list[0][1].keys()
    for key in metric_keys:
        avg_metric_dict[key] = 0.0
    tot_count = 0
    for (count, metric_dict) in metrics_list:
        tot_count += count
        for key in metric_keys:
            val = metric_dict[key]
            if type(val) == torch.tensor:
                val = val.item()
            avg_metric_dict[key] += count * val
    for key in metric_keys:
        avg_metric_dict[key] = avg_metric_dict[key] / tot_count

    # avg_metric_dict = {}

    # metric_keys = metrics_list[0][1].keys()
    # for key in metric_keys:
    #     avg_metric_dict[key] = {}
    #     for metric_name in metrics_list[0][1][key].keys():
    #         avg_metric_dict[key][metric_name] = 0.0
    # tot_count = 0
    # for (count, metric_dict) in metrics_list:
    #     tot_count += count
    #     for key in metric_keys:
    #         for metric_name in metric_dict[key].keys():
    #             val = metric_dict[key][metric_name]
    #             if type(val) == torch.tensor:
    #                 val = val.item()
    #             avg_metric_dict[key][metric_name] += (
    #                 count * val
    #             )
    # for key in metric_keys:
    #     for metric_name in metrics_list[0][1][key].keys():
    #         avg_metric_dict[key][metric_name] = (
    #             avg_metric_dict[key][metric_name] / tot_count
    #         )
    return avg_metric_dict


def euclidean_dist(w1, w2):
    dist = 0.0
    for key in w1.keys():
        dist += np.linalg.norm(w1[key] - w2[key]) ** 2
    return dist


# def compute_loss(model_trainer, client, train=True):

#     model_trainer.trainer.test(
#         model_trainer.model,
#         client.train_dataloader() if train else client.test_dataloader(),
#         verbose=False
#     )
#     return model_trainer.trainer.logged_metrics["test_loss"]


def compute_dist(
    trainer, model_1, model_2, client_1, client_2, dist_metric, tune=False
):
    if dist_metric == "euclidean":
        return euclidean_dist(model_1.get_model_wts(), model_2.get_model_wts())
    elif dist_metric == "cross_entropy":

        model_1_client_2 = 0.0
        for client in client_2:
            # if tune:
            #     trainer = pl.Trainer(
            #         enable_model_summary=False,
            #         enable_progress_bar=False,
            #         strategy=RayStrategy(num_workers=3, num_cpus_per_worker=3, use_gpu=True),
            #         precision=16,
            #         amp_backend="native",
            #         log_every_n_steps=1,
            #     )

            # else:
            #     trainer = pl.Trainer(
            #         devices=1,
            #         accelerator="gpu",
            #         enable_model_summary=False,
            #         enable_progress_bar=False,
            #         strategy="ddp_find_unused_parameters_false",
            #         precision=16,
            #         amp_backend="native",
            #         log_every_n_steps=1,
            #     )
            trainer.test(
                model_1.model, client.train_dataloader(), verbose=False, ckpt_path=None
            )
            model_1_client_2 += trainer.logged_metrics["test_loss"]
            # model_1_client_2 += compute_loss(model_1, client, train=True)
        model_1_client_2 = model_1_client_2 / len(client_2)
        model_2_client_1 = 0.0
        for client in client_1:
            # if tune:
            #     trainer = pl.Trainer(
            #         enable_model_summary=False,
            #         enable_progress_bar=False,
            #         strategy=RayStrategy(num_workers=3, num_cpus_per_worker=3, use_gpu=True),
            #         precision=16,
            #         amp_backend="native",
            #         log_every_n_steps=1,
            #     )
            # else:
            #     trainer = pl.Trainer(
            #         devices=1,
            #         accelerator="gpu",
            #         enable_model_summary=False,
            #         enable_progress_bar=False,
            #         strategy="ddp_find_unused_parameters_false",
            #         precision=16,
            #         amp_backend="native",
            #         log_every_n_steps=1,
            #     )
            trainer.test(
                model_2.model, client.train_dataloader(), verbose=False, ckpt_path=None
            )

            model_2_client_1 += trainer.logged_metrics["test_loss"]
            # model_2_client_1 += compute_loss(model_2, client, train=True)
        model_2_client_1 = model_2_client_1 / len(client_1)
        return (model_1_client_2 + model_2_client_1) / 2
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
    config = {}
    ## Config loaded from yaml file for the dataset and merged with the arguments from argparse
    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as err:
            print(err)
    # this updates args_dict by config with replacement
    config = args_dict | config
    config["path"] = {
        "data": os.path.join(config["path"], "seed_{}".format(config["seed"]), "data"),
        "results": os.path.join(
            config["path"],
            "seed_{}".format(config["seed"]),
            "results",
            args_dict["clustering"],
        ),
    }
    seed_everything(config["seed"], workers=True)

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


def get_device(config, i, cluster=False):
    if torch.cuda.device_count() == 1:
        return "cuda:0"
    else:
        raise ValueError(
            "Current implementation can handle only 1 GPU. {} GPUs were provided".format(
                torch.cuda.device_count()
            )
        )


def check_nan(metrics):
    for val in metrics.values():
        if np.isnan(np.array(val)).any() or np.isinf(np.array(val)).any():
            return True

    return False


def get_search_space(config):
    algo_config_path = os.path.join(
        "configs",
        "clustering",
        config["clustering"],
        config["dataset"]["name"],
    )
    hp_search_path = os.path.join(algo_config_path, "hp_config.py")
    spec = importlib.util.spec_from_file_location("module.name", hp_search_path)
    config_file = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = config_file
    spec.loader.exec_module(config_file)
    best_hp_path = os.path.join(algo_config_path, "best_hp.yaml")
    return best_hp_path, lambda trial: config_file.get_hp_config(trial, config)


def wt_dict_diff(wt_1, wt_2):
    assert wt_1.keys() == wt_2.keys(), "Both weight dicts have different keys"
    diff_dict = {}
    for key in wt_1.keys():
        diff_dict[key] = convert_to_cpu(wt_1[key]) - convert_to_cpu(wt_2[key])
    return diff_dict


def wt_dict_norm(wt):
    norm = 0
    for val in wt.values():
        norm += np.linalg.norm(convert_to_cpu(val)) ** 2
    return np.sqrt(norm)


def wt_dict_dot(wt_1, wt_2):
    assert wt_1.keys() == wt_2.keys(), "Both weight dicts have different keys"
    dot = 0.0
    for key in wt_1.keys():
        dot += np.dot(
            convert_to_cpu(wt_1[key]).reshape(-1), convert_to_cpu(wt_2[key]).reshape(-1)
        )
    return dot


def convert_to_cpu(val):
    val_arr = val
    if type(val) != np.ndarray:
        if val.device != "cpu":
            val_arr = val_arr.detach()
        val_arr = val_arr.detach()
    return val_arr


def compute_alpha_max(alpha_mat, partitions):

    keys = list(partitions.keys())
    return alpha_mat[partitions[keys[0]], :][:, partitions[keys[1]]].max()


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        data.update(cdata["user_data"])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def train_val_split(train_data, val_fraction=0.2):
    (X, y) = train_data
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    val_start_idx = int((1 - val_fraction) * X.shape[0])
    train_data = (X[:val_start_idx], y[:val_start_idx])
    val_data = (X[val_start_idx:], y[val_start_idx:])
    return train_data, val_data


def tune_config_update(config):
    if config["clustering"] == "sr_fca":
        config["refine"]["rounds"] = ceil(
            int(config["init"]["iterations"]) / int(config["refine"]["local_iter"])
        )

    elif config["clustering"] in ["ifca", "cfl", "fedavg"]:
        config["rounds"] = ceil(int(config["iterations"]) / int(config["local_iter"]))
    else:
        raise NotImplementedError(
            "Not implemented clustering {}".format(config["clustering"])
        )

    return config


def format_trainer_for_metrics(trainer, save_dir):
    trainer._default_root_dir = save_dir
    trainer.log_every_n_steps = 1
    trainer.limit_val_batches = 0
    trainer.enable_progress_bar = False
    trainer.enable_model_checkpointing = False
    trainer.limit_train_batches = 0
    trainer.callbacks = []
    # trainer.logger = None
    trainer.enable_model_checkpointing = False
    ## maybe some callbacks for formatting trainer for the main case.
    trainer._callback_connector.on_trainer_init(
        trainer.callbacks,
        trainer.checkpoint_callback,
        trainer.enable_model_checkpointing,
        trainer.enable_progress_bar,
        trainer.progress_bar_refresh_rate,
        trainer.process_position,
        trainer.default_root_dir,
        trainer.weights_save_path,
        trainer.enable_model_summary,
        trainer.weights_summary,
        trainer.stochastic_weight_avg,
        trainer.max_time,
        trainer.accumulate_grad_batches,
    )

    # hook
    trainer._call_callback_hooks("on_init_start")
    trainer._data_connector.on_trainer_init(
        trainer.check_val_every_n_epoch,
        trainer.reload_dataloaders_every_n_epochs,
        trainer.prepare_data_per_node,
    )

    trainer._init_debugging_flags(
        trainer.limit_train_batches,
        trainer.limit_val_batches,
        trainer.limit_test_batches,
        trainer.limit_predict_batches,
        trainer.val_check_interval,
        trainer.overfit_batches,
        trainer.fast_dev_run,
    )

    # Callback system
    trainer._call_callback_hooks("on_init_end")

    return trainer
