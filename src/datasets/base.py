import os
from src.datasets.synthetic import generate_synthetic_data
from src.datasets.simulated import load_simulated_dataset
from src.datasets.client import Client
from src.utils import set_seeds, read_data_config
from time import time


class FLDataset:
    def __init__(self, config, name=None, client_dict=None, tune=False):
        self.config = config
        if name is not None and client_dict is not None:
            self.name = name
            self.client_dict = client_dict
            set_seeds(self.config["seed"])
        else:
            self.name = config["dataset"]
            self.create_fldataset(tune=tune)

    def create_fldataset(self, tune):
        self.config = read_data_config(self.config)
        set_seeds(self.config["seed"])
        self.create_clients(tune=tune)

    def create_clients(self, tune):
        self.config["time"]["tdataset"] = time()
        train_chunks, test_chunks = get_dataset(self.config)
        self.config["time"]["tnew"] = time()
        print(
            "Time taken to get dataset : {} s".format(
                self.config["time"]["tnew"] - self.config["time"]["tdataset"]
            )
        )

        self.client_dict = {}
        for i in range(len(train_chunks)):

            self.client_dict[i] = Client(
                config=self.config,
                client_dataset=(train_chunks[i], test_chunks[i]),
                client_id=i,
                tune=tune,
            )
            if i == 0:
                self.config["time"]["tnew"] = time()
                print(
                    "Time taken to get 1 client : {} s".format(
                        self.config["time"]["tnew"] - self.config["time"]["tdataset"]
                    )
                )

        self.config["time"]["tnew"] = time()
        print(
            "Time taken to create clients : {} s".format(
                self.config["time"]["tnew"] - self.config["time"]["tdataset"]
            )
        )
        self.config["time"]["tdataset"] = self.config["time"]["tnew"]


# def custom_serializer(fldataset):
#     return (fldataset.config, fldataset.name, fldataset.client_dict)


# def custom_deserializer(config, name, client_dict):
#     return FLDataset(config=config, name=name, client_dict=client_dict)


def get_dataset(config):

    dataset_name = config["dataset"]["name"]
    if len(dataset_name.split("_")) > 1:
        client_het = dataset_name.split("_")[0]
        dataset_name = dataset_name.split("_")[-1]
    else:
        client_het = "real"
    dataset_path = config["path"]["data"]

    if dataset_name in ["mnist", "cifar10"]:
        train_chunks, test_chunks = load_simulated_dataset(
            dataset_name, client_het, dataset_path, config
        )
        ### Here log the distribution of clients to clusters
    elif dataset_name == "femnist":

        data_source = None
        raise NotImplementedError()

    elif dataset_name == "shakespeare":
        data_source = None
        raise NotImplementedError()

    elif dataset_name == "synthetic":
        train_chunks, test_chunks = generate_synthetic_data(config, dataset_path)

    else:
        raise ValueError("{} is not a valid dataset name".format(dataset_name))
    return train_chunks, test_chunks
