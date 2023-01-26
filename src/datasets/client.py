import os
from math import ceil, sqrt
from shutil import rmtree
from typing import List

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
# from ffcv.fields import FloatField, IntField, NDArrayField, RGBImageField
# from ffcv.fields.decoders import (
#     FloatDecoder,
#     IntDecoder,
#     NDArrayDecoder,
#     SimpleRGBImageDecoder,
# )
# from ffcv.loader import Loader, OrderOption
# from ffcv.pipeline.operation import Operation
# from ffcv.transforms import (
#     Convert,
#     Cutout,
#     RandomHorizontalFlip,
#     RandomTranslate,
#     ToDevice,
#     ToTensor,
#     ToTorchImage,
#     RandomResizedCrop
# )
# from ffcv.transforms.common import Squeeze
# from ffcv.writer import DatasetWriter
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.utils import get_device

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


class ClientWriteDataset(Dataset):
    def __init__(self, config, data, transform=None):
        super(ClientWriteDataset, self).__init__()
        self.config = config
        self.data = data[0]
        self.target = data[1]
        self.format_dataset_for_write()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx_data = self.data[idx]
        idx_target = self.target[idx]
        if self.config["dataset"]["name"] == "synthetic":
            idx_target = np.array(idx_target)
            idx_target = idx_target.astype("float32")
        if self.transform is not None:
            idx_data = self.transform(idx_data).float()
        
        if self.config["dataset"]["name"] in ["rot_mnist", "inv_mnist"]:
            idx_data = idx_data.float().squeeze().flatten()
        return idx_data, idx_target

    def format_dataset_for_write(self):
        if self.config["dataset"]["name"] in ["mnist","femnist" ]:
            self.data = self.data.reshape(-1, self.config["dataset"]["input_size"])
            if type(self.data) == torch.Tensor:
                self.data = self.data.float().numpy()
            elif type(self.data) == np.ndarray:
                self.data = self.data.astype("float32")
            else:
                raise ValueError(
                    "Invalid datatype of client features {}".format(type(self.data))
                )


class Client:
    def __init__(self, config, client_dataset, client_id, tune=False):

        train_data, test_data = client_dataset
        self.client_id = client_id
        if config["dataset"]["name"] == "rot_mnist":
            rot = 90*(client_id %4)
            transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation((rot,rot)), transforms.ToTensor()])
            train_transform = transform
            test_transform = transform
        # elif config["dataset"]["name"] == "rot_cifar10_ftrs":
        # #     train_transform = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN,CIFAR_STD)])
        # #     test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN,CIFAR_STD)])
        #     train_transform = transforms.Compose([transforms.ToTensor(),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(), transforms.Normalize(CIFAR_MEAN,CIFAR_STD)])
        #     test_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(224), transforms.Normalize(CIFAR_MEAN,CIFAR_STD)])
  
        else:
            train_transform=None
            test_transform=None
        train_writeset = ClientWriteDataset(config, train_data, transform=train_transform)
        test_writeset = ClientWriteDataset(config, test_data, transform=test_transform)
        
        # if config["dataset"]["name"] == "rot_cifar10_ftrs":
        #     train_writeset = get_resnet_ftrs(train_writeset, config, client_id)
        #     test_writeset = get_resnet_ftrs(test_writeset, config, client_id)
        # if config["dataset"]["name"].startswith("rot_cifar10"):
        #     temp_path = os.path.join(config["path"]["data"], "tmp_storage")
        #     os.makedirs(temp_path, exist_ok=True)
        #     train_beton_path = os.path.join(
        #         temp_path, "train_client_{}.beton".format(client_id)
        #     )
        #     test_beton_path = os.path.join(
        #         temp_path, "test_client_{}.beton".format(client_id)
        #     )
        #     (
        #         writer_pipeline,
        #         train_loader_pipeline,
        #         test_loader_pipeline,
        #     ) = get_pipelines(config, self.client_id)
        #     ## Issues with C code and rewrites when this is not always done.
        #     if not os.path.exists(train_beton_path) or not os.path.exists(
        #         test_beton_path
        #     ):

        #         train_writer = DatasetWriter(
        #             train_beton_path, writer_pipeline, num_workers=0
        #         )
        #         test_writer = DatasetWriter(
        #             test_beton_path, writer_pipeline, num_workers=0
        #         )
        #         train_writer.from_indexed_dataset(train_writeset)
        #         test_writer.from_indexed_dataset(test_writeset)

        #     self.trainloader = Loader(
        #         train_beton_path,
        #         batch_size=config["batch"]["train"],
        #         num_workers=8,
        #         order=OrderOption.QUASI_RANDOM,
        #         drop_last=False,
        #         pipelines=train_loader_pipeline,
        #     )
        #     self.testloader = Loader(
        #         test_beton_path,
        #         batch_size=config["batch"]["test"],
        #         num_workers=8,
        #         order=OrderOption.QUASI_RANDOM,
        #         drop_last=False,
        #         pipelines=test_loader_pipeline,
        #     )
        # else:
        self.trainloader = DataLoader(
            train_writeset,
            batch_size=config["batch"]["train"],
            shuffle=True,
            num_workers=0,
        )
        self.testloader = DataLoader(
            test_writeset,
            batch_size=config["batch"]["test"],
            shuffle=False,
            num_workers=0,
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
    if config["dataset"]["name"] == "synthetic":
        writer_pipeline = {
            "X": NDArrayField(
                shape=(config["dataset"]["dimension"],), dtype=np.dtype("float32")
            ),
            "y": IntField(),
        }
        train_loader_pipeline = {
            "X": [
                NDArrayDecoder(),
                ToTensor(),
                ToDevice(get_device(config, i)),
            ],
            "y": [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(get_device(config, i)),
                Convert(torch.float16),
            ],
        }
        test_loader_pipeline = train_loader_pipeline
    elif config["dataset"]["name"].endswith("mnist"):
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
                ToDevice(get_device(config, i)),
                Convert(torch.float16),
            ],
            "y": [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(get_device(config, i)),
                Convert(torch.float16),
            ],
        }
        test_loader_pipeline = train_loader_pipeline
    elif config["dataset"]["name"].endswith("cifar10"):
        writer_pipeline = {"X": RGBImageField(), "y": IntField()}
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice(get_device(config, i), non_blocking=True),
            Squeeze(),
        ]
        train_image_pipeline: List[Operation] = [
            SimpleRGBImageDecoder(),
            RandomHorizontalFlip(),
            RandomTranslate(padding=2),
            Cutout(8, tuple(map(int, CIFAR_MEAN))),
            ToTensor(),
            ToDevice(get_device(config, i)),
            ToTorchImage(),
            Convert(torch.float32),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]

        test_image_pipeline: List[Operation] = [
            SimpleRGBImageDecoder(),
            ToTensor(),
            ToDevice(get_device(config, i)),
            ToTorchImage(),
            Convert(torch.float16),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
        if i %2 ==1 :
            train_image_pipeline.extend([transforms.RandomRotation((180,180))])
            test_image_pipeline.extend([transforms.RandomRotation((180,180))])
    elif config["dataset"]["name"] == "rot_cifar10_ftrs":
        writer_pipeline = {"X": RGBImageField(), "y": IntField()}
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice(get_device(config, i), non_blocking=True),
            Squeeze(),
        ]
        train_image_pipeline: List[Operation] = [
            SimpleRGBImageDecoder(),
            # RandomHorizontalFlip(),
            # RandomTranslate(padding=2),
            Cutout(8, tuple(map(int, CIFAR_MEAN))),
            RandomResizedCrop(size=224, ratio=(0.75, 1.3333), scale=(0.08,1.0)),
            ToTensor(),
            ToDevice(get_device(config, i)),
            ToTorchImage(),
            
            Convert(torch.float32),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]

        test_image_pipeline: List[Operation] = [
            SimpleRGBImageDecoder(),
            RandomResizedCrop(size=224, scale=(1.0,1.0), ratio=(0.1,1.0)),
            ToTensor(),
            ToDevice(get_device(config, i)),
            ToTorchImage(),
            Convert(torch.float16),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
        if i %2 ==1 :
            train_image_pipeline.extend([transforms.RandomRotation((180,180))])
            test_image_pipeline.extend([transforms.RandomRotation((180,180))])

        train_loader_pipeline = {"X": train_image_pipeline, "y": label_pipeline}
        test_loader_pipeline = {"X": test_image_pipeline, "y": label_pipeline}
    elif config["dataset"]["name"] == "shakespeare":
        raise NotImplementedError
    else:
        raise NotImplementedError

    return writer_pipeline, train_loader_pipeline, test_loader_pipeline


def get_resnet_ftrs(writeset, config, client_id):
    return writeset
    num_classes = config["dataset"]["num_classes"]
    resnet_model = resnet18(pretrained=True)
    resnet_children = list(resnet_model.children())
    feature_model = nn.Sequential(*resnet_children[:-1])
    device = get_device(config, client_id)
    X = [], Y = []
    feature_model.eval()
    feature_model.to(device)
    with torch.no_grad():
        for img, label in tqdm(writeset):
            img = img.to(device)
            img_ftrs = feature_model(img.unsqueeze())
            
        