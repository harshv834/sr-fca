from time import time

from src.datasets.client import Client
from src.datasets.femnist import get_femnist
from src.datasets.shakespeare import get_shakespeare
from src.datasets.simulated import load_simulated_dataset
from src.datasets.synthetic import generate_synthetic_data
from src.utils import read_data_config, set_seeds
from tqdm import tqdm

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
        train_chunks, test_chunks = self.get_dataset()
        self.config["time"]["tnew"] = time()
        print(
            "Time taken to get dataset : {} s".format(
                self.config["time"]["tnew"] - self.config["time"]["tdataset"]
            )
        )

        self.client_dict = {}
        for i in tqdm(range(len(train_chunks))):

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

    def get_dataset(self):

        dataset_name = self.config["dataset"]["name"]
        if len(dataset_name.split("_")) > 1:
            client_het = dataset_name.split("_")[0]
            dataset_name = dataset_name.split("_")[1]
        else:
            client_het = "real"
        dataset_path = self.config["path"]["data"]

        if dataset_name in ["mnist", "cifar10"]:
            train_chunks, test_chunks = load_simulated_dataset(
                dataset_name, client_het, dataset_path, self.config
            )
            ### Here log the distribution of clients to clusters
        elif dataset_name == "femnist":
            train_chunks, test_chunks, client_idx = get_femnist(self.config)
            self.config["client_idx"] = client_idx

        elif dataset_name == "shakespeare":
            train_chunks, test_chunks, client_idx = get_shakespeare(self.config)
            self.config["client_idx"] = client_idx

        elif dataset_name == "synthetic":
            train_chunks, test_chunks = generate_synthetic_data(
                self.config, dataset_path
            )


        else:
            raise ValueError("{} is not a valid dataset name".format(dataset_name))
        return train_chunks, test_chunks
