import os
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from ffcv.fields import IntField, NDArrayField, RGBImageField
from ffcv.fields.decoders import (
    IntDecoder,
    NDArrayDecoder,
    SimpleRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    Convert,
    Cutout,
    RandomHorizontalFlip,
    RandomTranslate,
    ToTensor,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from src.utils import train_val_split

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


class Client(pl.LightningDataModule):
    def __init__(self, config, client_dataset, client_id, tune=False):
        super(Client, self).__init__()
        self.config = config
        train_data, test_data = client_dataset
        train_data, val_data = train_val_split(train_data)
        self.client_id = client_id
        self.train_writeset = ClientWriteDataset(config, train_data)
        self.val_writeset = ClientWriteDataset(config, val_data)
        self.test_writeset = ClientWriteDataset(config, test_data)

        if config["dataset"]["name"] == "rot_cifar10":
            temp_path = os.path.join(config["path"]["data"], "tmp_storage")
            # if os.path.exists(temp_path):
            #     rmtree(temp_path)
            os.makedirs(temp_path, exist_ok=True)
            self.train_beton_path = os.path.join(
                temp_path, "train_client_{}.beton".format(client_id)
            )
            self.val_beton_path = os.path.join(
                temp_path, "val_client_{}.beton".format(client_id)
            )
            self.test_beton_path = os.path.join(
                temp_path, "test_client_{}.beton".format(client_id)
            )
            (
                writer_pipeline,
                self.train_loader_pipeline,
                self.test_loader_pipeline,
            ) = get_pipelines(self.config, self.client_id)
            ## Issues with C code and rewrites when this is not always done.

            train_writer = DatasetWriter(
                self.train_beton_path, writer_pipeline, num_workers=1
            )
            val_writer = DatasetWriter(
                self.val_beton_path, writer_pipeline, num_workers=1
            )
            test_writer = DatasetWriter(
                self.test_beton_path, writer_pipeline, num_workers=1
            )
            train_writer.from_indexed_dataset(self.train_writeset)
            val_writer.from_indexed_dataset(self.val_writeset)
            test_writer.from_indexed_dataset(self.test_writeset)

    def train_dataloader(self):
        if self.config["dataset"]["name"] != "rot_cifar10":
            return DataLoader(
                self.train_writeset,
                batch_size=self.config["batch"]["train"],
                shuffle=True,
                num_workers=0,
            )
        else:
            return Loader(
                self.train_beton_path,
                batch_size=self.config["batch"]["train"],
                num_workers=0,
                order=OrderOption.SEQUENTIAL,
                drop_last=True,
                pipelines=self.train_loader_pipeline,
                # distributed=True,
                os_cache=True,
            )

    def test_dataloader(self):
        if self.config["dataset"]["name"] != "rot_cifar10":
            return DataLoader(
                self.test_writeset,
                batch_size=self.config["batch"]["test"],
                shuffle=False,
                num_workers=0,
            )
        else:
            return Loader(
                self.test_beton_path,
                batch_size=self.config["batch"]["test"],
                num_workers=0,
                order=OrderOption.SEQUENTIAL,
                drop_last=False,
                os_cache=True,
                pipelines=self.test_loader_pipeline,
                # distributed=True,
            )

    # TODO : Make this a loader
    def val_dataloader(self):
        if self.config["dataset"]["name"] != "rot_cifar10":
            return DataLoader(
                self.val_writeset,
                batch_size=self.config["batch"]["test"],
                shuffle=False,
                num_workers=0,
            )
        else:
            return Loader(
                self.val_beton_path,
                batch_size=self.config["batch"]["test"],
                num_workers=0,
                order=OrderOption.SEQUENTIAL,
                drop_last=False,
                os_cache=True,
                pipelines=self.test_loader_pipeline,
                # distributed=True,
            )

        # TODO : Add tune behaviour compatibility
        # if tune:
        #     self.train_iterator = None
        #     self.test_iterator = None
        # else:
        #     self.train_iterator = iter(self.trainloader)
        #     self.test_iterator = iter(self.testloader)

    # def sample_batch(self, train=True):
    #     if self.test_iterator is None and self.test_iterator is None:
    #         self.train_iterator = iter(self.trainloader)
    #         self.test_iterator = iter(self.testloader)
    #     iterator = self.train_iterator if train else self.test_iterator
    #     try:
    #         (X, y) = next(iterator)
    #     except StopIteration:
    #         if train:
    #             self.train_iterator = iter(self.trainloader)
    #             iterator = self.train_iterator
    #         else:
    #             self.test_iterator = iter(self.testloader)
    #             iterator = self.test_iterator
    #         (X, y) = next(iterator)
    #     return (X, y)


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
            "X": [NDArrayDecoder(), ToTensor()],
            "y": [IntDecoder(), ToTensor(), Squeeze()],
        }
        test_loader_pipeline = train_loader_pipeline
    elif config["dataset"]["name"].endswith("cifar10"):
        writer_pipeline = {"X": RGBImageField(), "y": IntField()}
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
        ]
        train_image_pipeline: List[Operation] = [
            SimpleRGBImageDecoder(),
            RandomHorizontalFlip(),
            RandomTranslate(padding=2),
            Cutout(8, tuple(map(int, CIFAR_MEAN))),
            ToTensor(),
            ToTorchImage(),
            Convert(torch.float32),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]

        test_image_pipeline: List[Operation] = [
            SimpleRGBImageDecoder(),
            ToTensor(),
            ToTorchImage(),
            Convert(torch.float32),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
        train_loader_pipeline = {"X": train_image_pipeline, "y": label_pipeline}
        test_loader_pipeline = {"X": test_image_pipeline, "y": label_pipeline}
    elif config["dataset"]["name"] == "shakespeare":
        raise NotImplementedError
    else:
        raise NotImplementedError

    return writer_pipeline, train_loader_pipeline, test_loader_pipeline
