from abc import ABC
from src.utils import read_algo_config
from math import ceil


class ClusterFLAlgo(ABC):
    def __init__(self, config, tune=False, tune_config=None):
        super(ClusterFLAlgo, self).__init__()
        self.cluster_map = {}
        self.tune = tune
        if self.tune:
            assert tune_config is not None, "No hyperparameter config provided"
            self.config = tune_config
            self.config["refine"]["rounds"] = ceil(
                int(self.config["init"]["iterations"])
                // int(self.config["refine"]["local_iter"])
            )
            # print("tune_config:", tune_config)
            # print("fldataset config", config)
            self.config = config | self.config

        else:
            self.config = read_algo_config(config)

    def cluster(self, experiment):
        raise NotImplementedError

    def empty_cluster_check(self, info):
        ## Zero client checking
        zero_client_clusters = [
            cluster_id
            for cluster_id, clients in self.cluster_map.items()
            if len(clients) == 0
        ]
        if len(zero_client_clusters) > 0:
            zero_client_cluster_names = ""
            for cluster_id in zero_client_clusters:
                zero_client_cluster_names += " {}".format(cluster_id)
            raise ValueError(
                "Clusters "
                + zero_client_cluster_names
                + " have 0 clients after "
                + info
            )