import os
import shutil
from time import time

import numpy as np
import torch

from src.clustering.base import TRIAL_MAP, ClusterFLAlgo
from src.trainers import ClientTrainer, ClusterTrainer
from src.utils import (
    avg_metrics,
    check_nan,
    vectorize_model_wts,
    unvectorize_model_wts,
)
from sklearn.cluster import KMeans
from tqdm import tqdm



class OneShotKMeans(ClusterFLAlgo):
    def __init__(self, config, tune=False, tune_config=None):
        super(OneShotKMeans, self).__init__(config, tune, tune_config)
        self.client_trainers = {
            i: ClientTrainer(config=self.config, client_id=i, mode="solo")
            for i in range(self.config["num_clients"])
        }

    def cluster(self, experiment):

        self.config["time"]["tcluster"] = time()
        self.local_train(experiment)
        self.config["time"]["tnew"] = time()
        print(
            "Time taken by Local Training step : {} s".format(
                self.config["time"]["tnew"] - self.config["time"]["tcluster"]
            )
        )
        self.config["time"]["tcluster"] = self.config["time"]["tnew"]
        if check_nan(self.local_metrics):
            raise ValueError("Nan or inf occurred in metrics")

        self.kmeans(experiment)
        self.config["time"]["tnew"] = time()
        print(
            "Time taken by KMeans on local model weights :{}".format(
                self.config["time"]["tnew"] - self.config["time"]["tcluster"]
            )
        )
        if check_nan(self.kmeans_metrics):
            raise ValueError("Nan or inf occurred in metrics")
        torch.save(
            self.cluster_map,
            os.path.join(self.config["path"]["results"], "cluster_map.pth"),
        )

        return self.kmeans_metrics

    def kmeans(self, experiment):
        

        kmeans_path = os.path.join(self.config["path"]["results"], "kmeans")
        kmeans_metrics = []

        client_model_wts = np.vstack(
            [
                vectorize_model_wts(trainer.model)
                for trainer in self.client_trainers.values()
            ]
        )
        self.kmeans_model = KMeans(
            n_clusters=self.config["num_clusters"],
            random_state=self.config["seed"],
            init="k-means++",
        )
        self.kmeans_model.fit(client_model_wts)
        self.cluster_map = {}
        self.cluster_trainers = {}
        kmeans_metrics = []

        for i in range(self.config["num_clusters"]):
            cluster_clients = np.where(self.kmeans_model.labels_ == i)[0].tolist()
            if len(cluster_clients) > 0:
                self.cluster_map[i] = cluster_clients
                cluster_center_wts = unvectorize_model_wts(
                    self.kmeans_model.cluster_centers_[i], self.client_trainers[0].model
                )
                cluster_trainer = ClusterTrainer(self.config, i)
                cluster_trainer.model.load_state_dict(cluster_center_wts)
                cluster_trainer.client_idx = cluster_clients
                cluster_trainer.set_save_dir(
                    os.path.join(kmeans_path, "cluster_{}".format(i))
                )
                cluster_trainer.set_save_dir(
                    os.path.join(kmeans_path, "cluster_{}".format(i))
                )
                kmeans_metrics.append(
                    (1, cluster_trainer.compute_metrics(experiment.client_dict))
                )
                self.cluster_trainers[i] = cluster_trainer
        self.kmeans_metrics = avg_metrics(kmeans_metrics)
        torch.save(self.kmeans_metrics, os.path.join(kmeans_path, "kmeans_metrics.pth"))
        return self.kmeans_metrics

    def local_train(self, experiment):

        client_dict = experiment.client_dict
        local_path = os.path.join(self.config["path"]["results"], "local_train")
        client_path = lambda i, path : os.path.join(path, "client_{}".format(i))
        if "from_init" in self.config.keys() and self.config["from_init"]:
            ## Get the path from init
            init_path = "/" + os.path.join(*(self.config["path"]["results"].split("/")[:-1] + ["sr_fca", "init"]))
            for i in tqdm(self.client_trainers.keys()):
                ## Set save directory for local models
                self.client_trainers[i].set_save_dir(client_path(i, local_path))
                model_path = os.path.join(client_path(i, init_path), "model.pth")
                ## Copy metrics and model of local model
                shutil.copy2(os.path.join(client_path(i, init_path), "metrics.pth"), os.path.join(client_path(i, local_path), "metrics.pth"))
                shutil.copy2(model_path, os.path.join(client_path(i, local_path), "model.pth"))
                
                ## Load saved model weights
                self.client_trainers[i].load_saved_weights()
            self.local_metrics = torch.load(os.path.join(init_path, "metrics.pth"))
            
        else:
            
            local_metrics = []
            for i in tqdm(self.client_trainers.keys()):
                self.client_trainers[i].set_save_dir(client_path(i, local_path))
                client_metrics = self.client_trainers[i].train(
                    client_data=client_dict[i],
                    local_iter=self.config["iterations"],
                )
                local_metrics.append((1, client_metrics))
            self.local_metrics = avg_metrics(local_metrics)
    
        torch.save(self.local_metrics, os.path.join(local_path, "metrics.pth"))

        # self.cluster_map = TRIAL_MAP
