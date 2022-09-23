from abc import ABC
from src.trainers import ClientTrainer
from src.utils import read_algo_config


class ClusterFLAlgo(ABC):
    def __init__(self, config):
        super(ClusterFLAlgo, self).__init__()
        self.cluster_map = {}
        self.config = read_algo_config(config)
        self.client_trainers = {
            i: ClientTrainer(config=self.config, client_id=i, mode="solo")
            for i in range(self.config["num_clients"])
        }

    def cluster(self, experiment):
        raise NotImplementedError
