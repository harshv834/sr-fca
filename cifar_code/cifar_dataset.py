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

import time

from typing import List

import torchvision
DATASET_LIB = {
    "mnist": torchvision.datasets.MNIST,
    "emnist": torchvision.datasets.EMNIST,
    "cifar10": torchvision.datasets.CIFAR10,
}

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

def split(dataset_size, num_clients, shuffle):
    split_idx = []
    all_idx = np.arange(dataset_size)
    if shuffle:
        all_idx = np.random.permutation(all_idx)
    split_idx = np.array_split(all_idx, num_clients)
    return split_idx


def dataset_split(train_data, test_data, num_clients, shuffle):
    train_size = train_data[0].shape[0]
    train_split_idx = split(train_size, num_clients, shuffle)
    train_chunks = [
        (
            train_data[0][train_split_idx[client].tolist()],
            np.array(train_data[1])[train_split_idx[client].tolist()].tolist(),
        )
        for client in range(num_clients)
    ]
    test_size = test_data[0].shape[0]
    test_split_idx = split(test_size, num_clients, shuffle)
    test_chunks = [
        (
            test_data[0][test_split_idx[client].tolist()],
            np.array(test_data[1])[test_split_idx[client].tolist()].tolist(),
        )
        for client in range(num_clients)
    ]
    return train_chunks, test_chunks


def make_client_datasets(config):
    train_chunks_total = []
    test_chunks_total = []
    train_dataset = DATASET_LIB[config["dataset"]](
        root=config["dataset_dir"], download=True, train=True
    )
    test_dataset = DATASET_LIB[config["dataset"]](
        root=config["dataset_dir"], download=True, train=False
    )

    train_data = (train_dataset.data, train_dataset.targets)
    test_data = (test_dataset.data, test_dataset.targets)
    for i in range(config["num_clusters"]):
        train_chunks, test_chunks = dataset_split(
            train_data, test_data, config["total_num_clients_per_cluster"], shuffle=True
        )
        chunks_idx = np.random.choice(
            np.arange(len(train_chunks)),
            size=config["num_clients_per_cluster"],
            replace=False,
        ).astype(int)
        train_chunks = [train_chunks[idx] for idx in chunks_idx]
        test_chunks = [test_chunks[idx] for idx in chunks_idx]
        # train_chunks = np.array(train_chunks)[chunks_idx].tolist()
        # test_chunks = np.array(test_chunks)[chunks_idx].tolist()
        train_chunks_total += train_chunks
        test_chunks_total += test_chunks
    return train_chunks_total, test_chunks_total


class ClientWriteDataset(Dataset):
    def __init__(self, data):
        super(ClientWriteDataset, self).__init__()
        self.data = data[0]
        self.labels = data[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx_data = self.data[idx]
        idx_labels = self.labels[idx]
        return (idx_data, idx_labels)


class Client:
    def __init__(
        self,
        train_data,
        test_data,
        client_id,
        train_image_pipeline,
        test_image_pipeline,
        label_pipeline,
        train_batch_size,
        test_batch_size,
        save_dir,
    ):
        train_writeset = ClientWriteDataset(train_data)
        test_writeset = ClientWriteDataset(test_data)
        temp_path = os.path.join(save_dir, "tmp_storage")
        os.makedirs(temp_path, exist_ok=True)
        train_beton_path = os.path.join(
            temp_path, "train_client_{}.beton".format(client_id)
        )
        test_beton_path = os.path.join(
            temp_path, "test_client_{}.beton".format(client_id)
        )
        ## Write data to beton if it isn't already present
        if not os.path.exists(train_beton_path) and os.path.exists(test_beton_path):
            train_writer = DatasetWriter(
                train_beton_path,
                {"image": RGBImageField(), "label": IntField()},
            )
            test_writer = DatasetWriter(
                test_beton_path,
                {"image": RGBImageField(), "label": IntField()},
            )
            train_writer.from_indexed_dataset(train_writeset)
            test_writer.from_indexed_dataset(test_writeset)
        
        self.client_id = client_id
        self.trainloader = Loader(
            train_beton_path,
            batch_size=train_batch_size,
            num_workers=8,
            order=OrderOption.QUASI_RANDOM,
            drop_last=True,
            pipelines={"image": train_image_pipeline, "label": label_pipeline},
        )
        self.testloader = Loader(
            test_beton_path,
            batch_size=test_batch_size,
            num_workers=8,
            order=OrderOption.QUASI_RANDOM,
            drop_last=False,
            pipelines={"image": test_image_pipeline, "label": label_pipeline},
        )

        # self.trainloader = DataLoader(
        #     self.trainset, batch_size=train_batch_size, shuffle=True, num_workers=8
        # )
        # self.testloader = DataLoader(
        #     self.testset, batch_size=test_batch_size, shuffle=False, num_workers=8
        # )
        self.train_iterator = iter(self.trainloader)
        self.test_iterator = iter(self.testloader)
        self.save_dir = os.path.join(save_dir, "init", "client_{}".format(client_id))

    def sample_batch(self, train=True):
        iterator = self.train_iterator if train else self.test_iterator
        try:
            (data, labels) = next(iterator)
        except StopIteration:
            if train:
                self.train_iterator = iter(self.trainloader)
                iterator = self.train_iterator
            else:
                self.test_iterator = iter(self.testloader)
                iterator = self.test_iterator
            (data, labels) = next(iterator)
        return (data, labels)
    
    
