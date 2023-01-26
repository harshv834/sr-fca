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

import torchvision

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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed", type=int, required=True, help="Random seed for the experiment"
)




class Mul(nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
    )


class ResNet9(nn.Module):
    def __init__(self, NUM_CLASSES=10):
        super(ResNet9, self).__init__()
        self.model = nn.Sequential(
            conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
            conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
            conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            Residual(nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
            conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
            nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            nn.Linear(128, NUM_CLASSES, bias=False),
            Mul(0.2),
        )

    def forward(self, x):
        return self.model(x)


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

    def train(self, client_data_list, lr_schedule):
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

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
        scaler = GradScaler()

        for iteration in range(self.config["iterations"]):
            t0 = time.time()
            trmean_buffer = {}
            for idx, param in self.model.named_parameters():
                trmean_buffer[idx] = []
            train_loss = 0
            optimizer.zero_grad(set_to_none=True)
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

            # if (
            #     iteration % self.config["save_freq"] == 0
            #     or iteration == self.config["iterations"] - 1
            # ):
            #     self.save_model_weights()
            #     self.save_metrics(train_loss_list, test_acc_list, iteration)

            # if iteration % self.config["print_freq"] == 0:
            #     print(
            #         "Iteration : {} \n , Train Loss : {} \n, Test Acc : {} \n, Time : {}\n".format(
            #             iteration, train_loss, test_acc, time_taken
            #         )
            #     )

        self.model.eval()
        # self.model.cpu()
        return train_loss_list, test_acc_list

    def test(self, client_data_list):
        self.load_model_weights()
        self.model.eval()
        self.model.to(self.device)
        test_acc = 0
        for client_data in client_data_list:
            test_acc += calc_acc(self.model, self.device, client_data, train=False)
        self.model.cpu()
        return test_acc


def determine_clustering(ifca_trainers, client_data_list):

    cluster_map = {0: [], 1: []}
    for i in range(16):
        loss_list = []
        for j in range(2):
            loss_list.append(
                calc_loss(
                    ifca_trainers[j].model, "cuda:0", client_data_list[i], train=True
                )
            )
        if loss_list[0] > loss_list[1]:
            cluster_map[1].append(i)
        else:
            cluster_map[0].append(i)
    return cluster_map


def main(args):

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
    config["optimizer_params"] = {"lr": 0.5, "momentum": 0.9, "weight_decay": 5e-4}
    config["save_freq"] = 400
    config["print_freq"] = 400
    config["model"] = "resnet9"
    config["optimizer"] = "sgd"
    config["loss_func"] = "cross_entropy"
    # config["model_params"] = {"num_channels": 1 , "num_classes"  : 62}
    config["model_params"] = {}
    config["device"] = torch.device("cuda:0")

    rounds = 2400

    # rounds = 2400 // 10
    # local_iter = 10
    config["iterations"] = 1
    # import ipdb;ipdb.set_trace()
    
    ## Initial cluster_map
    all_clients = list(range(16))
    np.random.shuffle(all_clients)
    cluster_map = {0: all_clients[:8], 1: all_clients[8:]}


    # cluster_map = {0: [0, 2, 4, 6, 8, 10, 12, 14], 1: [1, 3, 5, 7, 9, 11, 13, 15]}


    client_loaders = np.array(client_loaders)
    ifca_trainers = [
        GlobalTrainer(config, os.path.join(config["results_dir"], "ifca", "cluster_0")),
        GlobalTrainer(
            config,
            os.path.join(
                config["results_dir"],
                "ifca",
                "cluster_1",
            ),
        ),
    ]
    iters_per_epoch = 50000 // int(
        config["train_batch"] * config["total_num_clients_per_cluster"]
    )
    epochs = 2400 // iters_per_epoch
    lr_schedule = np.interp(
        np.arange((epochs + 1) * iters_per_epoch),
        [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
        [0, 1, 0],
    )
    metrics = {0:{"train_loss" : [], "test_acc":  []}, 1:{"train_loss" : [], "test_acc" : []}}
    for round_id in tqdm(range(rounds)):
        for i in range(2):
            if len(cluster_map[i]) > 0:
                train_loss_list, test_acc_list = ifca_trainers[i].train(
                    client_loaders[cluster_map[i]],
                    lr_schedule[
                        round_id: round_id + 2
                    ],
                )
                metrics[i]["train_loss"] += train_loss_list
                metrics[i]["test_acc"] += test_acc_list
            # else:
            #     print("Cluster {} has 0 clients in round {}".format(i, round_id))
        cluster_map = determine_clustering(ifca_trainers, client_loaders)
        if round_id % config["print_freq"] == 0 or round_id == rounds - 1:
            print("Curr cluster_map : {}".format(cluster_map))
            print("Curr test metrics : cluster 0 : {}, cluster  1 : {}".format(metrics[0]["test_acc"][-1], metrics[1]["test_acc"][-1]))
            
        if round_id % config["save_freq"] == 0 or round_id == rounds - 1:

            torch.save(
                cluster_map,
                os.path.join(
                    config["results_dir"], "ifca", "cluster_map.pth"
                ),
            )
            torch.save(metrics, os.path.join(config["results_dir"], "ifca", "metrics.pth"))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
