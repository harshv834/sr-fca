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
import pickle

import time

from typing import List
from collections import OrderedDict
import torchvision
from sklearn.cluster import KMeans
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
from ffcv.writer import DatasetWriter
import itertools

from cifar_dataset import DATASET_LIB, Client, make_client_datasets

from .model import ResNet9

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed", type=int, required=True, help="Random seed for the experiment"
)
parser.add_argument(
    "--from_init",
    action=argparse.BooleanOptionalAction  
)




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

    def load_saved_weights(self):
        """Load saved model weights

        Raises:
            ValueError: when no saved weights present.
        """

        model_path = os.path.join(self.save_dir, "model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
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



class ClientTrainer(BaseTrainer):
    def __init__(self, config, save_dir, client_id):
        super(ClientTrainer, self).__init__(config, save_dir)
        self.client_id = client_id

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, client_data):
        train_loss_list = []
        test_acc_list = []
        self.model.to(memory_format=torch.channels_last).cuda()
        # (self.device)
        self.model.train()
        optimizer = OPTIMIZER_LIST[self.config["optimizer"]](
            self.model.parameters(), **self.config["optimizer_params"]
        )
        iters_per_epoch = 50000 // int(
            self.config["train_batch"] * self.config["total_num_clients_per_cluster"]
        )
        epochs = self.config["iterations"] // iters_per_epoch
        lr_schedule = np.interp(
            np.arange((epochs + 1) * iters_per_epoch),
            [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
            [0, 1, 0],
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
        scaler = GradScaler()

        for iteration in tqdm(range(self.config["iterations"])):
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            (X, Y) = client_data.sample_batch(train=True)
            with autocast():
                out = self.model(X)
                loss = self.loss_func(out, Y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss = loss.detach().cpu().numpy().item()
            train_loss_list.append(train_loss)
            test_acc = calc_acc(self.model, self.device, client_data, train=False)
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

            if (
                iteration % self.config["print_freq"] == 0
                or iteration == self.config["iterations"] - 1
            ):
                print(
                    "Iteration : {} \n , Train Loss : {} \n, Test Acc : {} \n, Time : {}\n".format(
                        iteration, train_loss, test_acc, time_taken
                    )
                )

        self.model.eval()
        self.model.cpu()

    def test(self, client_data):
        self.load_model_weights()
        self.model.eval()
        self.model.to(self.device)
        acc = calc_acc(self.model, client_data)
        self.model.cpu()
        return acc


MODEL_LIST = {"resnet9": ResNet9}
OPTIMIZER_LIST = {"sgd": optim.SGD, "adam": optim.Adam}
LOSSES = {"cross_entropy": nn.CrossEntropyLoss(label_smoothing=0.1)}


def calc_loss(model, client_data, train):
    loader = client_data.trainloader if train else client_data.testloader
    model.eval()
    model.cuda()

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



def vectorize_model_wts(model):
    """Flatten all model weights into single vector

    Args:
        model (nn.Module): Model whose weights need to be flattened

    Returns:
        np.ndarray: 1D vector of flattened model weigts.
    """
    model_wts = model.state_dict()
    wt_list = [
        wt.numpy().flatten()
        for wt in list(model_wts.values())
    ]
    wt_list = [wt for wt in wt_list if np.issubdtype(wt.dtype, np.integer) or np.issubdtype(wt.dtype, np.floating)]
    vectorized_wts = np.hstack(wt_list)
    return vectorized_wts


def unvectorize_model_wts(flat_wts, model):
    """Convert flattened model weights to an ordered state dict of the model

    Args:
        flat_wts (np.ndarray): 1D array with the flattened weights
        model (torch.nn.Module): Model whose state dict format we need to adhere to

    Returns:
        OrderedDict: Format flat_wts according to state dict of model
    """
    model_wts = model.state_dict()
    model_wts_to_update = OrderedDict(
        {
            key: val
            for key, val in model_wts.items()
            if np.issubdtype(val.numpy().dtype, np.integer) or np.issubdtype(val.numpy().dtype, np.floating)
        }
    )
    flat_tensor_len = [val.flatten().shape[0] for val in model_wts_to_update.values()]
    start_count = 0
    for i, (key, val) in enumerate(model_wts_to_update.items()):
        end_count = start_count + flat_tensor_len[i]
        flat_tensor = flat_wts[start_count:end_count]
        model_wts_to_update[key] = torch.tensor(
            flat_tensor.reshape(val.shape), dtype=val.dtype
        )
    model_wts_to_update = model_wts | model_wts_to_update
    return model_wts_to_update


class ClusterTrainer(BaseTrainer):
    def __init__(self, config, save_dir, cluster_id):
        super(ClusterTrainer, self).__init__(config, save_dir)
        self.cluster_id = cluster_id

    def train(self, client_data_list):
        num_clients = len(client_data_list)

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

        iters_per_epoch = 50000 * 0.5 // int(
            self.config["train_batch"] * self.config["total_num_clients_per_cluster"]
        )
        epochs = self.config["iterations"] // iters_per_epoch
        lr_schedule = np.interp(
            np.arange((epochs + 1) * iters_per_epoch),
            [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
            [0, 1, 0],
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
        scaler = GradScaler()

        for iteration in tqdm(range(self.config["iterations"])):
            t0 = time.time()
            trmean_buffer = {}
            for idx, param in self.model.named_parameters():
                trmean_buffer[idx] = []
            train_loss = 0
            for client in client_data_list:
                optimizer.zero_grad(set_to_none=True)
                (X, Y) = client.sample_batch()
                loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
                with autocast():
                    out = self.model(X)
                    loss = loss_func(out, Y)
                scaler.scale(loss).backward()
                train_loss += loss.detach().cpu().numpy().item()
                with torch.no_grad():
                    for idx, param in self.model.named_parameters():
                        trmean_buffer[idx].append(param.grad.clone())

            optimizer.zero_grad(set_to_none=True)
            start_idx = int(self.config["beta"] * num_clients)
            end_idx = int((1 - self.config["beta"]) * num_clients)
            if end_idx <= start_idx + 1:
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

            if iteration % self.config["print_freq"] == 0 or iteration  == self.config["iterations"] - 1:
                print(
                    "Iteration : {} \n , Train Loss : {} \n, Test Acc : {} \n, Time : {}\n".format(
                        iteration, train_loss, test_acc, time_taken
                    )
                )

        self.model.eval()
        # self.model.cpu()

    def test(self, client_data_list):
        self.load_model_weights()
        self.model.eval()
        self.model.to(self.device)
        test_acc = 0
        for client_data in client_data_list:
            test_acc += calc_acc(self.model, self.device, client_data, train=False)
        self.model.cpu()
        return test_acc


# class GlobalTrainer(BaseTrainer):
#     def __init__(self,  config, save_dir):
#         super(GlobalTrainer, self).__init__(config, save_dir)

#     def train(self, client_data_list):
#         num_clients = len(client_data_list)

#         train_loss_list = []
#         test_acc_list = []
#         self.model.to(memory_format = torch.channels_last).cuda()
#         self.model.train()
#         optimizer = OPTIMIZER_LIST[self.config["optimizer"]](self.model.parameters(), **self.config["optimizer_params"])

#         iters_per_epoch = 50000//int(self.config['train_batch']*self.config['total_num_clients_per_cluster'])
#         epochs = self.config["iterations"]// iters_per_epoch
#         lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
#                         [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
#                         [0, 1, 0])
#         scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
#         scaler = GradScaler()

#         for iteration in tqdm(range(self.config["iterations"])):
#             t0 = time.time()
#             trmean_buffer = {}
#             for idx, param in self.model.named_parameters():
#                 trmean_buffer[idx] = []
#             train_loss = 0
#             optimizer.zero_grad(set_to_none=True)
#             for client in client_data_list:
#                 optimizer.zero_grad(set_to_none=True)
#                 (X,Y) = client.sample_batch()
#                 loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
#                 with autocast():
#                     out = self.model(X)
#                     loss = loss_func(out,Y)
#                 scaler.scale(loss).backward()
#                 train_loss += loss.detach().cpu().numpy().item()
#                 with torch.no_grad():
#                     for idx, param in self.model.named_parameters():
#                         trmean_buffer[idx].append(param.grad.clone())

#             optimizer.zero_grad(set_to_none=True)
#             start_idx = 0
#             end_idx = num_clients


#             for idx, param in self.model.named_parameters():
#                 sorted, _  = torch.sort(torch.stack(trmean_buffer[idx], dim=0), dim=0)
#                 new_grad = sorted[start_idx:end_idx,...].mean(dim=0)
#                 param.grad = new_grad
#                 trmean_buffer[idx] = []
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()
#             train_loss = train_loss/num_clients
#             train_loss_list.append(train_loss)
#             test_acc = 0
#             for client_data in client_data_list:
#                 test_acc += calc_acc(self.model, self.device, client_data, train=False)
#             test_acc = test_acc/num_clients
#             test_acc_list.append(test_acc)
#             self.model.train()
#             t1 = time.time()
#             time_taken = t1 - t0

#             if iteration % self.config["save_freq"] == 0 or iteration == self.config["iterations"] - 1:
#                 self.save_model_weights()
#                 self.save_metrics(train_loss_list, test_acc_list, iteration)

#             if iteration% self.config["print_freq"] == 0:
#                 print("Iteration : {} \n , Train Loss : {} \n, Test Acc : {} \n, Time : {}\n".format(iteration,  train_loss, test_acc, time_taken))

#         self.model.eval()
#         self.model.cpu()


#     def test(self, client_data_list):
#         self.load_model_weights()
#         self.model.eval()
#         self.model.to(self.device)
#         test_acc = 0
#         for client_data in client_data_list:
#             test_acc += calc_acc(self.model, self.device, client_data, train=False)
#         self.model.cpu()
#         return test_acc


args = parser.parse_args()
config = {}
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
config["dataset"] = "cifar10"
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
                save_dir=config["results_dir"],
            )
        )

# model = model.to(memory_format=torch.channels_last).cuda()

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
        
# import ipdb;ipdb.set_trace()
kmeans_path = os.path.join(config["results_dir"], "kmeans")
kmeans_metrics = []

client_model_wts = np.vstack(
    [
        vectorize_model_wts(trainer.model)
        for trainer in client_trainers
    ]
)
kmeans_model = KMeans(
    n_clusters=config["num_clusters"],
    random_state=config["seed"],
    init="k-means++",
)
kmeans_model.fit(client_model_wts)
cluster_map = {}
cluster_trainers = {}
kmeans_metrics = []
# import ipdb;ipdb.set_trace()
test_acc = 0.0

for i in range(config["num_clusters"]):
    cluster_clients = np.where(kmeans_model.labels_ == i)[0].tolist()
    if len(cluster_clients) > 0:
        cluster_test_acc = 0.0
        cluster_map[i] = cluster_clients
        cluster_center_wts = unvectorize_model_wts(
            kmeans_model.cluster_centers_[i], client_trainers[0].model
        )
        cluster_trainer = ClusterTrainer(config, os.path.join(kmeans_path, "cluster_{}".format(i)), i)
        cluster_trainer.model.load_state_dict(cluster_center_wts)
        cluster_trainer.model.to(memory_format=torch.channels_last).cuda()
        for client_idx in cluster_clients:
            test_acc += calc_acc(cluster_trainer.model, config["device"], client_loaders[client_idx], train=False)
        cluster_trainers[i] = cluster_trainer
test_acc = test_acc/16
torch.save({"test_acc" : test_acc}, os.path.join(kmeans_path, "metrics.pth"))
torch.save(cluster_map, os.path.join(kmeans_path,  "cluster_map.pth"))
print("Final Test acc : {}".format(test_acc))


