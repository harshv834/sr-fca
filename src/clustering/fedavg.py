import os
from math import ceil
from time import time

from src.clustering.base import ClusterFLAlgo
from src.trainers import ClusterTrainer
from src.utils import check_nan


class FedAvg(ClusterFLAlgo):
    def __init__(self, config, tune=False, tune_config=None):
        super(FedAvg, self).__init__(config, tune, tune_config)
        if tune:
            self.config["rounds"] = ceil(
                self.config["iterations"] / self.config["local_iter"]
            )
        self.fedavg_path = os.path.join(self.config["path"]["results"])

        self.fedavg_trainer = ClusterTrainer(self.config, 0)
        self.fedavg_trainer.set_save_dir(self.fedavg_path)

    def cluster(self, experiment):
        self.config["time"]["tcluster"] = time()
        client_dict = experiment.client_dict
        metrics = self.fedavg_trainer.train(
            client_dict=client_dict,
            client_idx=range(self.config["num_clients"]),
            local_iter=self.config["local_iter"],
            rounds=self.config["rounds"],
        )
        if check_nan(metrics):
            raise ValueError("Nan or inf occurred in metrics")
        self.metrics = metrics
        return metrics
