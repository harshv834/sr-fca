import logging
import os
from src.datasets.synthetic import generate_synthetic_data
from src.datasets.simulated import load_simulated_dataset
from src.datasets.client import Client
from src.utils import set_seeds, read_data_config


class FLDataset:
    def __init__(self, args):
        self.config = vars(args)
        self.name = args.dataset
        self.clients = self.create_fldataset()

    def create_fldataset(self):
        self.config = read_data_config(self.config)
        set_seeds(self.config["seed"])
        self.clients = self.create_clients()

    def create_clients(self):
        train_chunks, test_chunks = zip(get_dataset(self.config))
        self.client_dict = {
            i: Client(self.config["batch"], dataset, i)
            for (i, dataset) in enumerate(zip(train_chunks, test_chunks))
        }


def get_dataset(config):
    dataset_name = config["name"]
    if len(dataset_name.split("_")) > 1:
        dataset_name = config["name"].split("_")[-1]
        client_het = config["name"].split("_")[0]
    else:
        client_het = "real"
    dataset_path = config["path"]["data"]

    if dataset_name in ["mnist", "cifar10"]:
        train_chunks, test_chunks = load_simulated_dataset(
            dataset_name, client_het, dataset_path, config
        )
        ### Here log the distribution of clients to clusters
        logging.info()
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
