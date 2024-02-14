import torch
import numpy as np
from torch.utils.data import Dataset
from abc import ABC
from tqdm import tqdm
import torchvision
import os
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from cifar_dataset import DATASET_LIB, Client, make_client_datasets
import time

from typing import List
import itertools
import torchvision
from collections import OrderedDict
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    RandomHorizontalFlip,
    Cutout,
    RandomTranslate,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze

from cifar_utils import wt_dict_diff,compute_alpha_max,wt_dict_dot,wt_dict_norm, calc_acc, calc_loss

from model import ResNet9


parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed", type=int, required=True, help="Random seed for the experiment"
)
parser.add_argument(
    "--gamma_max", type=float, default = 0.5, help="gamma max for cfl"
)
parser.add_argument("--client_threshold", type=float, default = 0.1, help = "client threshold for cfl")
parser.add_argument("--stop_threshold", type=float, default = 0.1, help = "stop threshold for cfl")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help = "debug variable")


cluster_map = {}
cluster_trainers = {}
cluster_metrics = {}
cluster_idx_to_train_queue = []


# def calc_acc(model, device, client_data, train):
#     loader = client_data.trainloader if train else client_data.testloader
#     model.eval()
#     with torch.no_grad():
#         total_correct, total_num = 0.0, 0.0
#         for ims, labs in loader:
#             with autocast():
#                 out = model(ims)  # Test-time augmentation
#                 total_correct += out.argmax(1).eq(labs).sum().cpu().item()
#                 total_num += ims.shape[0]

#     return total_correct * 100.0 / total_num


# def calc_loss(model, device, client_data, train):
#     loader = client_data.trainloader if train else client_data.testloader
#     model.eval()

#     loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
#     with torch.no_grad():
#         total_loss, total_num = 0.0, 0.0
#         for ims, labs in loader:
#             with autocast():
#                 out = model(ims)  # Test-time augmentation
#                 total_loss += loss_func(out, labs).detach().item()
#                 # total_correct += out.argmax(1).eq(labs).sum().cpu().item()
#                 total_num += ims.shape[0]

#     return total_loss / total_num


class BaseTrainer(ABC):
    def __init__(self, config, save_dir):
        super(BaseTrainer, self).__init__()
        self.model = MODEL_LIST[config["model"]](**config["model_params"])
        self.save_dir = save_dir
        self.device = config["device"]
        self.loss_func = LOSSES[config["loss_func"]]
        self.config = config
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def load_model_weights(self):
        model_path = os.path.join(self.save_dir, "model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            print("No model present at path : {}".format())

    def save_model_weights(self):
        model_path = os.path.join(self.save_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)

    def save_metrics(self, train_loss, test_acc, iteration):
        torch.save(
            {"train_loss": train_loss, "test_acc": test_acc},
            os.path.join(self.save_dir, "metrics_{}.pkl".format(iteration)),
        )




MODEL_LIST = {"resnet9": ResNet9}
OPTIMIZER_LIST = {"sgd": optim.SGD, "adam": optim.Adam}
LOSSES = {"cross_entropy": nn.CrossEntropyLoss(label_smoothing=0.1)}


class GlobalTrainer(BaseTrainer):
    def __init__(self, config, save_dir):
        super(GlobalTrainer, self).__init__(config, save_dir)

    def get_model_wts(self):
        wts = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                wts[name] = param.data
        return wts

    def train(self, client_data_list):
        num_clients = len(client_data_list)
        iters_per_epoch = (50000*num_clients//16) // int(
            self.config["train_batch"] * self.config["total_num_clients_per_cluster"]
        )
        epochs = 2400 // iters_per_epoch
        lr_schedule = np.interp(
            np.arange((epochs + 1) * iters_per_epoch),
            [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
            [0, 1, 0],
        )

        train_loss_list = []
        test_acc_list = []
        self.model.to(memory_format=torch.channels_last).cuda()
        self.model.train()
        optimizer = OPTIMIZER_LIST[self.config["optimizer"]](
            self.model.parameters(), **self.config["optimizer_params"]
        )
        # eff_num_workers = int(num_clients/(1 - 2*beta))
        # if eff_num_workers > 0:
        #     eff_batch_size = self.config["train_batch"]/eff_num_workers
        #     for i in range(num_clients):
        #         client_data_list[i].trainloader.batch_size = eff_batch_size

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
        scaler = GradScaler()

        for iteration in tqdm(range(self.config["iterations"])):
            model_wt = self.get_model_wts()
            if iteration > 1:
                if self.config["stop_threshold"] > 0:
                    diff_wt = wt_dict_norm(wt_dict_diff(model_wt, self.prev_model_wt))
            # return wt_dict_norm(diff_wt) < self.stop_threshold
                    if  diff_wt < self.config["stop_threshold"]:
                        iteration = self.config["iterations"]-1

            if self.config["debug"] and iteration > 2:
                iteration = self.config["iterations"]-1
            t0 = time.time()
            trmean_buffer = {}
            # if iteration + 1 == self.config["iterations"]:
            self.prev_model_wt = self.get_model_wts()
            for idx, param in self.model.named_parameters():
                trmean_buffer[idx] = []
            train_loss = 0
            optimizer.zero_grad(set_to_none=True)
            self.client_wt_diff = []
            for client in client_data_list:
                optimizer.zero_grad(set_to_none=True)
                (X, Y) = client.sample_batch()
                loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
                with autocast():
                    out = self.model(X)
                    loss = loss_func(out, Y)
                scaler.scale(loss).backward()
                train_loss += loss.detach().cpu().numpy().item()
                if iteration + 1 == self.config["iterations"]:
                    client_grad_dict = {}                


                with torch.no_grad():
                    for idx, param in self.model.named_parameters():
                        trmean_buffer[idx].append(param.grad.clone())
                        
                        if iteration + 1 == self.config["iterations"]:
                            client_grad_dict[idx]  = param.grad.clone()
                            
                if iteration + 1 == self.config["iterations"]:
                    self.client_wt_diff.append(client_grad_dict)
            optimizer.zero_grad(set_to_none=True)
            start_idx = 0
            end_idx = num_clients

            for idx, param in self.model.named_parameters():
                sorted, _ = torch.sort(torch.stack(trmean_buffer[idx], dim=0), dim=0)
                new_grad = sorted[start_idx:end_idx, ...].mean(dim=0)
                param.grad = new_grad
                trmean_buffer[idx] = []
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss = train_loss / num_clients
            train_loss_list.append(train_loss)
            test_acc = 0
            for client_data in client_data_list:
                test_acc += calc_acc(self.model, self.device, client_data, train=False)
            test_acc = test_acc / num_clients
            test_acc_list.append(test_acc)
            self.model.train()
            t1 = time.time()
            time_taken = t1 - t0

            if (
                iteration % self.config["save_freq"] == 0
                or iteration == self.config["iterations"] - 1
            ):
                self.save_model_weights()
                self.save_metrics(train_loss_list, test_acc_list, iteration)

            if iteration % self.config["print_freq"] == 0 or iteration == self.config["iterations"] -1:
                print(
                    "Iteration : {} \n , Train Loss : {} \n, Test Acc : {} \n, Time : {}\n".format(
                        iteration, train_loss, test_acc, time_taken
                    )
                )
            if iteration == self.config["iterations"] - 1:
                break
        self.model.eval()
        # self.model.cpu()
        return {"train_loss": train_loss_list[-1], "test_acc": test_acc_list[-1]}

    def test(self, client_data_list):
        self.load_model_weights()
        self.model.eval()
        self.model.to(self.device)
        test_acc = 0
        for client_data in client_data_list:
            test_acc += calc_acc(self.model, self.device, client_data, train=False)
        self.model.cpu()
        return test_acc



def cfl_single_node(config, client_dict, cluster_id):
    ## Train a model for the cluster
    cluster_save_dir = os.path.join(
        config["cluster_path"], "cluster_{}".format(cluster_id)
    )

    cluster_trainer = GlobalTrainer(
        config, cluster_save_dir
    )
    cluster_metrics[cluster_id] = cluster_trainer.train(
        client_data_list=list(client_dict.values()),
    )
    global cluster_trainers 
    global cluster_map
    global cluster_idx_to_train_queue

    cluster_trainers[cluster_id] = cluster_trainer

    ## Split cluster into two parts
    if (
        len(cluster_map[cluster_id]) == 1
        or len(cluster_map) == config["num_clusters"]
    ):
        return
    else:
        ## Compute alpha for every client pair in cluster

        alpha_mat, max_loss_client = compute_alpha_mat(cluster_id)
        if np.isnan(alpha_mat).any() or np.isinf(alpha_mat).any():
            print(
                "Nan or inf occurred in alpha for cluster : {}".format(cluster_id)
            )
        ## Obtain optimal bipartitioning to maximize
        partitions = optimal_bipartitioning(cluster_id, alpha_mat)

        ## Obtain max alpha between two partitions
        alpha_max_cross = compute_alpha_max(
            alpha_mat, partitions, cluster_map[cluster_id]
        )
        if (
            max_loss_client >= config["client_threshold"]
            and np.sqrt((1 - alpha_max_cross) / 2) > config["gamma_max"]
        ) or True:
            _ = cluster_map.pop(cluster_id)
            _ = cluster_trainers.pop(cluster_id)
            for key, val in partitions.items():
                cluster_map[key] = val
                cluster_idx_to_train_queue.append(key)
        return

def compute_alpha_mat(cluster_id):
    """Compute alpha matrix which is cosine similarity of loss gradient of different clients at optima for the cluster.

    Args:
        cluster_id (_type_): _description_

    Returns:
        _type_: final alpha matric
    """

    ## Cosine similarity is 1 if the two clients are same, so start
    ## with an identity matrix
    client_idx = cluster_map[cluster_id]
    alpha_mat = np.diag(np.ones(len(client_idx)))
    ## Compute weight differences/loss gradient at optima for clients in given cluster
    client_wt_diff = cluster_trainers[cluster_id].client_wt_diff 
    wt_diff_norms = {i: wt_dict_norm(client_wt_diff[i]) for i in range(len(client_idx))}
    for (i, j) in itertools.combinations(range(len(client_idx)), 2):
        if (
            wt_diff_norms[client_idx[i]] < 1e-10
            or wt_diff_norms[client_idx[j]] < 1e-10
        ):
            alpha_mat[i][j] = 0
        else:
            dot = wt_dict_dot(
                client_wt_diff[client_idx[i]], client_wt_diff[client_idx[j]]
            )
            alpha_mat[i][j] = dot / (
                wt_diff_norms[client_idx[i]] * wt_diff_norms[client_idx[j]]
            )
        alpha_mat[j][i] = alpha_mat[i][j]
    max_loss_client = max(wt_diff_norms.values())
    return alpha_mat, max_loss_client

def optimal_bipartitioning(cluster_id, alpha_mat):
    client_idx = cluster_map[cluster_id]
    num_clients = len(client_idx)
    alpha_flat = alpha_mat.flatten()
    sorted_idx = (-1 * alpha_flat).argsort()
    C = {i: set([i]) for i in client_idx}
    cluster_list = list(C.keys())
    for i in range(num_clients**2):
        i_1 = client_idx[sorted_idx[i] // num_clients]
        i_2 = client_idx[sorted_idx[i] % num_clients]
        c_temp = set([])
        j_min = max(cluster_list)
        for j in cluster_list:
            if i_1 in C[j] or i_2 in C[j]:
                j_min = min(j, j_min)
                c_temp = c_temp.union(C[j])
                C[j] = set()
        C[j_min] = c_temp
        cluster_list = []
        for key in C.keys():
            if len(C[key]) > 0:
                cluster_list.append(key)

        C = {j: C[j] for j in cluster_list}
        if len(cluster_list) == 2:
            partition_1_id = (cluster_id + 1) * 2
            partition_2_id = (cluster_id + 1) * 2 + 1
            return {
                partition_1_id: [client_id for client_id in C[cluster_list[0]]],
                partition_2_id: [client_id for client_id in C[cluster_list[1]]],
            }

    # def cluster(self, experiment):
    #     """Main method to create clusters of clients

    #     Args:
    #         experiment (dict): Dict of client data used for the experiment

    #     Raises:
    #         ValueError: When Nan or inf appears in metrics

    #     Returns:
    #         dict: Metrics of trained cluster federated learning 
    #     """
    #     ### Initialize the client dict and put all clients inside the first cluster which has cluster_id 0
    #     self.config["time"]["tcluster"] = time()

    #     client_dict = experiment.client_dict
    #     init_cluster_id = 0
    #     self.cluster_map = {init_cluster_id: list(range(self.config["num_clients"]))}

    #     ## Add this cluster_id to a FIFO queue which contains cluster_idx to train next
    #     self.cluster_idx_to_train.append(init_cluster_id)

    #     ## While required number of clusters haven't been trained, perform CFL on a new cluster id
    #     while len(self.cluster_trainers.keys()) < self.config["num_clusters"]:
    #         ## Put cluster_idx to train in a queue and pop the queue and train each cluster.
    #         if len(self.cluster_idx_to_train) > 0:
    #             cluster_idx_to_train = self.cluster_idx_to_train.pop(0)
    #             client_dict_to_train = {
    #                 client_idx: client_dict[client_idx]
    #                 for client_idx in self.cluster_map[cluster_idx_to_train]
    #             }
    #             self.cfl_single_node(client_dict_to_train, cluster_idx_to_train)
    #         else:
    #             break
    #     ## Among the final clusters which remain,
    #     self.metrics = []
    #     for cluster_id in self.cluster_map.keys():
    #         self.cluster_trainers[cluster_id].client_idx = self.cluster_map[cluster_id]
    #         metrics = self.cluster_trainers[cluster_id].compute_metrics(client_dict)
    #         if check_nan(metrics):
    #             raise ValueError("Nan or inf occurred in metrics")
    #         self.metrics.append((len(self.cluster_map[cluster_id]), metrics))
    #     self.metrics = avg_metrics(self.metrics)
    #     torch.save(self.metrics, os.path.join(self.config["path"]["results"], "metrics.pth"))
    #     torch.save(self.cluster_map, os.path.join(self.config["path"]["results"], "cluster_map.pth"))
    #     return self.metrics


def main(args):

    config = args
    # config["seed"] = args.seed
    seed = config["seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)

    config["participation_ratio"] = 0.5
    config["total_num_clients_per_cluster"] = 16
    config["num_clients_per_cluster"] = int(
        config["participation_ratio"] * config["total_num_clients_per_cluster"]
    )
    config["num_clusters"] = 2
    config["num_clients"] = config["num_clients_per_cluster"] * config["num_clusters"]
    config["dataset"] = "cifar10"
    ## Changed path to match existing results.
    config["dataset_dir"] = "../experiments/dataset"
    config["results_dir"] = "../experiments/results"
    config["results_dir"] = os.path.join(
        config["results_dir"], config["dataset"], "seed_{}".format(seed)
    )

    train_dataset = DATASET_LIB[config["dataset"]](
        root=config["dataset_dir"], download=True, train=True
    )
    test_dataset = DATASET_LIB[config["dataset"]](
        root=config["dataset_dir"], download=True, train=False
    )

    os.makedirs(config["results_dir"], exist_ok=True)
    train_chunks, test_chunks = make_client_datasets(config)

    client_loaders = []

    config["train_batch"] = 100
    config["test_batch"] = 512
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]

    for i in tqdm(range(config["num_clusters"])):
        for j in tqdm(range(config["num_clients_per_cluster"])):
            idx = i * config["num_clients_per_cluster"] + j

            label_pipeline: List[Operation] = [
                IntDecoder(),
                ToTensor(),
                ToDevice("cuda:0"),
                Squeeze(),
            ]
            train_image_pipeline: List[Operation] = [
                SimpleRGBImageDecoder(),
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(int, CIFAR_MEAN))),
                ToTensor(),
                ToDevice("cuda:0", non_blocking=True),
                ToTorchImage(),
                Convert(torch.float16),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]

            test_image_pipeline: List[Operation] = [
                SimpleRGBImageDecoder(),
                ToTensor(),
                ToDevice("cuda:0", non_blocking=True),
                ToTorchImage(),
                Convert(torch.float16),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]

            if i > 0:
                train_image_pipeline.extend([transforms.RandomRotation((180, 180))])
                test_image_pipeline.extend([transforms.RandomRotation((180, 180))])

            client_loaders.append(
                Client(
                    train_chunks[idx],
                    test_chunks[idx],
                    idx,
                    train_image_pipeline=train_image_pipeline,
                    test_image_pipeline=test_image_pipeline,
                    label_pipeline=label_pipeline,
                    train_batch_size=config["train_batch"],
                    test_batch_size=config["test_batch"],
                    save_dir=os.path.join(config["results_dir"]),
                )
            )

    # model = model.to(memory_format=torch.channels_last).cuda()

    # config["save_dir"] = os.path.join("./results")
    config["iterations"] = 2400
    # config["debug"] = args.debug
    config["optimizer_params"] = {"lr": 0.5, "momentum": 0.9, "weight_decay": 5e-4}
    config["save_freq"] = 400
    config["print_freq"] = 400
    config["model"] = "resnet9"
    config["optimizer"] = "sgd"
    config["loss_func"] = "cross_entropy"
    # config["model_params"] = {"num_channels": 1 , "num_classes"  : 62}
    config["model_params"] = {}
    config["device"] = torch.device("cuda:0")
    # config["client_threshold"]  =args.client_threshold
    # config["gamma_max"] = args.gamma_max

    rounds = 2400

    # rounds = 2400 // 10
    # local_iter = 10
    # config["iterations"] = 1
    
    
    config["cluster_path"] = os.path.join(config["results_dir"], "cfl", "clusters")

    global cluster_map 
    global cluster_trainers
    global cluster_metrics
    global cluster_idx_to_train_queue

    
    
    init_cluster_id = 0
    cluster_map = {init_cluster_id: list(range(config["num_clients"]))}

    ## Add this cluster_id to a FIFO queue which contains cluster_idx to train next
    cluster_idx_to_train_queue.append(init_cluster_id)

    ## While required number of clusters haven't been trained, perform CFL on a new cluster id
   
    while len(cluster_trainers.keys()) < config["num_clusters"]:
        ## Put cluster_idx to train in a queue and pop the queue and train each cluster.
        if len(cluster_idx_to_train_queue) > 0:
            cluster_idx_to_train = cluster_idx_to_train_queue.pop(0)
            client_dict_to_train = {
                client_idx: client_loaders[client_idx]
                for client_idx in cluster_map[cluster_idx_to_train]
            }
            cfl_single_node(config, client_dict_to_train, cluster_idx_to_train)
        else:
            break
    ## Among the final clusters which remain,
    test_acc = 0.0
    for cluster_id in cluster_map.keys():
        cluster_test_acc = 0.0
        for client_idx in cluster_map[cluster_id]:
            cluster_test_acc += calc_acc(cluster_trainers[cluster_id].model, config["device"], client_loaders[client_idx], train=False)
        test_acc += cluster_test_acc
    test_acc  = test_acc/16
    torch.save({"test_acc" : test_acc}, os.path.join(config["results_dir"], "cfl", "metrics.pth"))
    torch.save(cluster_map, os.path.join(config["results_dir"],"cfl",  "cluster_map.pth"))
    print("Test Acc : {}".format(test_acc))
    return test_acc
    
    
    
    
    # ## Initial cluster_map
    # all_clients = list(range(16))
    # np.random.shuffle(all_clients)
    # cluster_map = {0: all_clients[:8], 1: all_clients[8:]}


    # # cluster_map = {0: [0, 2, 4, 6, 8, 10, 12, 14], 1: [1, 3, 5, 7, 9, 11, 13, 15]}


    # client_loaders = np.array(client_loaders)
    # ifca_trainers = [
    #     GlobalTrainer(config, os.path.join(config["results_dir"], "ifca", "cluster_0")),
    #     GlobalTrainer(
    #         config,
    #         os.path.join(
    #             config["results_dir"],
    #             "ifca",
    #             "cluster_1",
    #         ),
    #     ),
    # ]
    # iters_per_epoch = 50000 // int(
    #     config["train_batch"] * config["total_num_clients_per_cluster"]
    # )
    # epochs = 2400 // iters_per_epoch
    # lr_schedule = np.interp(
    #     np.arange((epochs + 1) * iters_per_epoch),
    #     [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
    #     [0, 1, 0],
    # )
    # metrics = {0:{"train_loss" : [], "test_acc":  []}, 1:{"train_loss" : [], "test_acc" : []}}
    # for round_id in tqdm(range(rounds)):
    #     for i in range(2):
    #         if len(cluster_map[i]) > 0:
    #             train_loss_list, test_acc_list = ifca_trainers[i].train(
    #                 client_loaders[cluster_map[i]],
    #                 lr_schedule[
    #                     round_id: round_id + 2
    #                 ],
    #             )
    #             metrics[i]["train_loss"] += train_loss_list
    #             metrics[i]["test_acc"] += test_acc_list
    #         # else:
    #         #     print("Cluster {} has 0 clients in round {}".format(i, round_id))
    #     cluster_map = determine_clustering(ifca_trainers, client_loaders)
    #     if round_id % config["print_freq"] == 0 or round_id == rounds - 1:
    #         print("Curr cluster_map : {}".format(cluster_map))
    #         print("Curr test metrics : cluster 0 : {}, cluster  1 : {}".format(metrics[0]["test_acc"][-1], metrics[1]["test_acc"][-1]))
            
    #     if round_id % config["save_freq"] == 0 or round_id == rounds - 1:

    #         torch.save(
    #             cluster_map,
    #             os.path.join(
    #                 config["results_dir"], "ifca", "cluster_map.pth"
    #             ),
    #         )
    #         torch.save(metrics, os.path.join(config["results_dir"], "ifca", "metrics.pth"))


if __name__ == "__main__":
    args = parser.parse_args()
    args = vars(args)

    main(args)
