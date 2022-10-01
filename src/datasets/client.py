import os
from math import ceil, sqrt
from shutil import rmtree
from typing import List

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from ffcv.fields import FloatField, IntField, NDArrayField, RGBImageField
from ffcv.fields.decoders import (FloatDecoder, IntDecoder, NDArrayDecoder,
                                  SimpleRGBImageDecoder)
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (Convert, Cutout, RandomHorizontalFlip,
                             RandomTranslate, ToDevice, ToTensor, ToTorchImage)
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
from torch.utils.data import DataLoader, Dataset

from src.utils import get_device

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


class ClientWriteDataset(Dataset):
    def __init__(self, config, data):
        super(ClientWriteDataset, self).__init__()
        self.config = config
        self.data = data[0]
        self.target = data[1]
        self.format_dataset_for_write()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx_data = self.data[idx]
        idx_target = self.target[idx]
        if self.config["dataset"]["name"] == "synthetic":
            idx_target = np.array(idx_target)
            idx_target = idx_target.astype("float32")
        return idx_data, idx_target

    def format_dataset_for_write(self):
        if self.config["dataset"]["name"].endswith("mnist"):
            self.data = self.data.reshape(-1, self.config["dataset"]["input_size"])
            if type(self.data) == torch.Tensor:
                self.data = self.data.float().numpy()
            elif type(self.data) == np.ndarray:
                self.data = self.data.astype("float32")
            else:
                raise ValueError(
                    "Invalid datatype of client features {}".format(type(self.data))
                )


# class ClientDataset(Dataset):
#     def __init__(self, data, transforms=None):
#         super(ClientDataset, self).__init__()
#         self.data = data[0]
#         self.labels = data[1]
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         idx_data = self.data[idx]
#         if self.transforms is not None:
#             transformed_data = self.transforms(idx_data)
#         else:
#             transformed_data = idx_data
#         idx_labels = self.labels[idx]
#         return (transformed_data, idx_labels)


# # 1,000,000 inputs each of dimension 10,000 = 40GB of data
# N, D = 100, 1000
# X = np.random.rand(N, D).astype("float32")
# # Ground-truth vector
# W, b = np.random.rand(D).astype("float32"), np.random.rand()
# # Response variables
# Y = X @ W + np.random.randn(N)


# class LinearRegressionDataset:
#     def __getitem__(self, idx):
#         return (X[idx], np.array(Y[idx]).astype("float32"))

#     def __len__(self):
#         return len(X)


class Client:
    def __init__(self, config, client_dataset, client_id, tune=False):

        train_data, test_data = client_dataset
        self.client_id = client_id
        train_writeset = ClientWriteDataset(config, train_data)
        test_writeset = ClientWriteDataset(config, test_data)

        if config["dataset"]["name"] not in ["synthetic", "shakespeare"]:

            temp_path = os.path.join(config["path"]["data"], "tmp_storage")
            if os.path.exists(temp_path):
                rmtree(temp_path)
            os.makedirs(temp_path, exist_ok=True)
            train_beton_path = os.path.join(
                temp_path, "train_client_{}.beton".format(client_id)
            )
            test_beton_path = os.path.join(
                temp_path, "test_client_{}.beton".format(client_id)
            )
            (
                writer_pipeline,
                train_loader_pipeline,
                test_loader_pipeline,
            ) = get_pipelines(config, self.client_id)
            ## Issues with C code and rewrites when this is not always done.

            train_writer = DatasetWriter(
                train_beton_path, writer_pipeline, num_workers=1
            )
            test_writer = DatasetWriter(test_beton_path, writer_pipeline, num_workers=1)
            train_writer.from_indexed_dataset(train_writeset)
            test_writer.from_indexed_dataset(test_writeset)

            self.trainloader = Loader(
                train_beton_path,
                batch_size=config["batch"]["train"],
                num_workers=1,
                order=OrderOption.QUASI_RANDOM,
                drop_last=True,
                pipelines=train_loader_pipeline,
            )
            self.testloader = Loader(
                test_beton_path,
                batch_size=config["batch"]["test"],
                num_workers=1,
                order=OrderOption.QUASI_RANDOM,
                drop_last=False,
                pipelines=test_loader_pipeline,
            )
        else:
            self.trainloader = DataLoader(
                train_writeset,
                batch_size=config["batch"]["train"],
                shuffle=True,
                num_workers=1,
            )
            self.testloader = DataLoader(
                test_writeset,
                batch_size=config["batch"]["test"],
                shuffle=False,
                num_workers=1,
            )

        if tune:
            self.train_iterator = None
            self.test_iterator = None
        else:
            self.train_iterator = iter(self.trainloader)
            self.test_iterator = iter(self.testloader)

    def sample_batch(self, train=True):
        if self.test_iterator is None and self.test_iterator is None:
            self.train_iterator = iter(self.trainloader)
            self.test_iterator = iter(self.testloader)
        iterator = self.train_iterator if train else self.test_iterator
        try:
            (X, y) = next(iterator)
        except StopIteration:
            if train:
                self.train_iterator = iter(self.trainloader)
                iterator = self.train_iterator
            else:
                self.test_iterator = iter(self.testloader)
                iterator = self.test_iterator
            (X, y) = next(iterator)
        return (X, y)


def get_pipelines(config, i):
    # if config["dataset"]["name"].endswith("mnist"):
    #     writer_pipeline = {
    #         "X": NDArrayField(
    #             shape=(config["dataset"]["dimension"],), dtype=np.dtype("float32")
    #         ),
    #         "y": NDArrayField(shape=(1,), dtype=np.dtype("float32")),
    #     }
    #     train_loader_pipeline = {
    #         "X": [
    #             NDArrayDecoder(),
    #             ToTensor(),
    #             ToDevice(torch.device(get_device(config, i)), non_blocking=True),
    #         ],
    #         "y": [
    #             NDArrayDecoder(),
    #             ToTensor(),
    #             Squeeze(),
    #             ToDevice(torch.device(get_device(config, i)), non_blocking=True),
    #         ],
    #     }
    #     test_loader_pipeline = train_loader_pipeline
    if config["dataset"]["name"].endswith("mnist"):
        writer_pipeline = {
            "X": NDArrayField(
                shape=(config["dataset"]["input_size"],), dtype=np.dtype("float32")
            ),
            "y": IntField(),
        }
        train_loader_pipeline = {
            "X": [
                NDArrayDecoder(),
                ToTensor(),
                ToDevice(torch.device(get_device(config, i)), non_blocking=True),
                Convert(torch.float16),
            ],
            "y": [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(torch.device(get_device(config, i)), non_blocking=True),
            ],
        }
        test_loader_pipeline = train_loader_pipeline
    elif config["dataset"]["name"].endswith("cifar10"):
        writer_pipeline = {"X": RGBImageField(), "y": IntField()}
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
        train_loader_pipeline = {"X": train_image_pipeline, "y": label_pipeline}
        test_loader_pipeline = {"X": test_image_pipeline, "y": label_pipeline}
    elif config["dataset"]["name"] == "shakespeare":
        raise NotImplementedError
    else:
        raise NotImplementedError

    return writer_pipeline, train_loader_pipeline, test_loader_pipeline
