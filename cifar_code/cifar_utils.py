import torch
import os
import pickle
from torch.cuda.amp import autocast
import torch.nn as nn
import numpy as np
import random
import argparse

def calc_local_acc_from_old(base_path):
    init_path = os.path.join(base_path, "init")
    test_acc = 0.0
    for i in range(16):
        metrics = torch.load(os.path.join(init_path, "node_{}".format(i), "metrics_2399.pkl"))
        test_acc += metrics['test_acc'][-1]
    test_acc = test_acc/16
    return test_acc


# def calc_sr_fca_acc_from_old(base_path):
#     refine_path = os.path.join(base_path, "refine_0")
#     with open(os.path.join(refine_path, "cluster_maps.pkl"), "rb") as f:
#         cluster_maps = pickle.load(f)
    
#     test_acc = 0.0
#     for i in range(16):
#         metrics = torch.load(os.path.join(init_path, "node_{}".format(i), "metrics_2399.pkl"))
#         test_acc += metrics['test_acc'][-1]
#     test_acc = test_acc/16
#     return test_acc


def calc_acc(model, device, client_data, train):
    loader = client_data.trainloader if train else client_data.testloader
    model.eval()
    with torch.no_grad():
        total_correct, total_num = 0.0, 0.0
        for ims, labs in loader:
            with autocast():
                out = model(ims)  # Test-time augmentation
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]

    return total_correct * 100.0 / total_num


def calc_loss(model, device, client_data, train):
    loader = client_data.trainloader if train else client_data.testloader
    model.eval()

    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
    with torch.no_grad():
        total_loss, total_num = 0.0, 0.0
        for ims, labs in loader:
            with autocast():
                out = model(ims)  # Test-time augmentation
                total_loss += loss_func(out, labs).detach().item()
                # total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]

    return total_loss / total_num


def avg_metrics(metrics_list):

    avg_metric_dict = {}

    metric_keys = metrics_list[0][1].keys()
    for key in metric_keys:
        avg_metric_dict[key] = {}
        for metric_name in metrics_list[0][1][key].keys():
            avg_metric_dict[key][metric_name] = 0.0
    tot_count = 0
    for (count, metric_dict) in metrics_list:
        tot_count += count
        for key in metric_keys:
            for metric_name in metric_dict[key].keys():
                avg_metric_dict[key][metric_name] += (
                    count * metric_dict[key][metric_name]
                )
    for key in metric_keys:
        for metric_name in metrics_list[0][1][key].keys():
            avg_metric_dict[key][metric_name] = (
                avg_metric_dict[key][metric_name] / tot_count
            )
    return avg_metric_dict

def check_nan(metrics):
    for key in metrics.keys():
        for val in metrics[key].values():
            if np.isnan(np.array(val)).any() or np.isinf(np.array(val)).any():
                return True

    return False

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
            val_arr = val_arr.cpu()
        val_arr = val_arr.cpu()
    return val_arr


def compute_alpha_max(alpha_mat, partitions, client_idx):

    keys = list(partitions.keys())
    return alpha_mat[[client_idx.index(el) for el in partitions[keys[0]]], :][
        :, [client_idx.index(el) for el in partitions[keys[1]]]
    ].max()


def cross_entropy_metric(trainer_1, trainer_2, client_1, client_2):
    trainer_1_client_2 = 0.0
    for client in client_2:
        trainer_1_client_2 += calc_loss(trainer_1.model, client, train=True)
        trainer_1_client_2 = trainer_1_client_2 / len(client_2)
        trainer_2_client_1 = 0.0
        for client in client_1:
            trainer_2_client_1 += calc_loss(trainer_2.model, client, train=True)
        trainer_2_client_1 = trainer_2_client_1 / len(client_1)
        return (trainer_1_client_2 + trainer_2_client_1) / 2


def model_weights_diff(w_1, w_2):
    norm_sq = 0
    assert w_1.keys() == w_2.keys(), "Model weights have different keys"
    for key in w_1.keys():
        if w_1[key].dtype == torch.float32:
            norm_sq += (w_1[key] - w_2[key]).norm() ** 2
    return np.sqrt(norm_sq)


def create_config(args):
    config =  vars(args)
    config["seed"] = args.seed
    seed = config["seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)


    ### Define beta here for cluster trainer
    config["beta"] = 0

    config["participation_ratio"] = 0.5
    config["total_num_clients_per_cluster"] = 16
    config["num_clients_per_cluster"] = int(
        config["participation_ratio"] * config["total_num_clients_per_cluster"]
    )
    config["num_clusters"] = 2
    config["num_clients"] = config["num_clients_per_cluster"] * config["num_clusters"]
    config["dataset"] = "cifar10_" + args.het
    config["dataset_dir"] = "../experiments/dataset"
    config["results_dir"] = "../experiments/results"
    config["results_dir"] = os.path.join(
        config["results_dir"], config["dataset"], "seed_{}".format(seed)
    )


    os.makedirs(config["results_dir"], exist_ok=True)



    config["train_batch"] = 100
    config["test_batch"] = 512


    # config["save_dir"] = os.path.join("./results")
    config["iterations"] = 2400
    config["optimizer_params"] = {"lr": 0.5, "momentum": 0.9, "weight_decay": 5e-4}
    config["save_freq"] = 400
    config["print_freq"] = 400
    config["model"] = "resnet9"
    config["optimizer"] = "sgd"
    config["loss_func"] = "cross_entropy"
    # config["model_params"] = {"num_channels": 1 , "num_classes"  : 62}
    config["model_params"] = {}
    config["device"] = torch.device("cuda:0")
    
    return config



def create_argparser():
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
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help = "debug variable")

    return parser
