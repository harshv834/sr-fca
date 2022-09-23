from src.models.base import MODEL_DICT
from abc import ABC
import torch.nn as nn
import os
import torch
from torch.cuda.amp import autocast, GradScaler
from src.utils import compute_metric, get_optimizer, get_lr_scheduler, avg_metrics
from tqdm import tqdm
import logging
import random


class BaseTrainer(ABC):
    def __init__(self, config):
        super(BaseTrainer, self).__init__()
        self.config = config
        self.model = MODEL_DICT[self.config["model"]]
        if self.config["name"] == "synthetic":
            self.loss_func = nn.MSELoss()
        else:
            label_smoothing = (
                self.config["label_smoothing"]
                if "label_smoothing" in self.config.keys()
                else 0.0
            )
            self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def load_model_weights(self):
        """Load saved model weights

        Raises:
            ValueError: when no saved weights present.
        """
        model_path = os.path.join(self.save_dir, "model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise ValueError("No model present at path : {}".format())
        self.model.eval()

    def save_model_weights(self):
        """Save model weights"""
        if self.save_dir is not None:
            model_path = os.path.join(self.save_dir, "model.pth")
            torch.save(self.model.state_dict(), model_path)

    def save_metrics(self):
        if self.metrics is not None:
            torch.save(
                self.metrics,
                os.path.join(
                    self.save_dir,
                    "metrics.pth",
                ),
            )

    def compute_loss(self, client_data, train=True):
        return compute_metric(self.model, client_data, train, loss=self.loss_func)

    def compute_acc(self, client_data, train=True):
        return compute_metric(self.model, client_data, train)

    def compute_metrics(self, client_data):
        return {
            "train": {
                "acc": compute_metric(self.model, client_data, train=True),
                "loss": compute_metric(
                    self.model, client_data, train=False, loss=self.loss_func
                ),
            },
            "test": {
                "loss": compute_metric(
                    self.model, client_data, train=False, loss=self.loss_func
                ),
                "acc": compute_metric(self.model, client_data, train=False),
            },
        }

    def test(self, client_data):
        self.load_model_weights()
        metrics = self.compute_metrics(client_data)
        self.model.cpu()
        return metrics


class ClientTrainer(BaseTrainer):
    def __init__(self, config, client_id, mode="fed"):
        super(ClientTrainer, self).__init__(config)
        self.client_id = client_id
        self.mode = mode
        if self.mode not in ["fed", "solo"]:
            raise ValueError("Invalid mode for ClientTrainer {}".format(self.mode))

    def set_save_dir(self, save_dir):
        if self.mode == "solo":
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)

    def train(self, client_data, local_iter, round=None):
        if self.mode == "solo":
            self.metrics = {}
        else:
            metrics = None

        self.model = self.model.to(memory_format=torch.channels_last).cuda()
        self.model.train()
        optimizer = get_optimizer(self.model.parameters(), self.config)
        scheduler = get_lr_scheduler(self.config, optimizer, local_iter, round)
        scaler = GradScaler()

        for iteration in tqdm(range(local_iter)):
            optimizer.zero_grad(set_to_none=True)
            (X, Y) = client_data.sample_batch(train=True)
            with autocast():
                out = self.model(X)
                loss = self.loss_func(out, Y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ## Local Freq
            if self.mode == "solo":
                if (
                    iteration % self.config["local_freq"] == 0
                    or iteration == local_iter - 1
                ):
                    metrics = self.compute_metrics(client_data)
                    self.save_model_weights()
                    self.save_metrics()
                    self.metrics[iteration] = metrics
                    logging.info(
                        "Iteration : {} \n , Train -- Loss : {} , Acc : {}, \n Test-- Loss : {}, Acc : {} \n, Time :\n".format(
                            iteration,
                            metrics["train"]["loss"],
                            metrics["train"]["acc"],
                            metrics["test"]["loss"],
                            metrics["test"]["loss"],
                        )
                    )
        self.model.eval()
        self.model.cpu()
        return metrics


class ClusterTrainer(BaseTrainer):
    def __init__(self, config, cluster_id):
        super(ClusterTrainer, self).__init__(config)
        self.cluster_id = cluster_id

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, client_dict, client_idx, local_iter, rounds):
        self.num_clients = len(client_idx)
        self.client_idx = client_idx
        client_trainers = {
            i: ClientTrainer(config=self.config, client_id=i, mode="fed")
            for i in client_idx
        }
        self.metrics = {}
        for round_id in tqdm(range(rounds)):
            metrics = []
            if (
                self.config["num_clients_per_round"] >= len(client_idx)
                and self.config["clustering"] == "ifca"
            ):
                selected_clients = random.sample(
                    client_idx, self.config["num_clients_per_round"]
                )
            client_trainers = self.send_to_clients(client_trainers, selected_clients)
            for i in selected_clients:
                _ = client_trainers[i].train(client_dict[i], local_iter, round_id)
            self.average_model(client_trainers, selected_clients)
            if round_id % self.config["global_freq"] == 0 or round_id == rounds - 1:
                metrics = self.compute_metrics(client_dict)
                self.save_model_weights()
                self.save_metrics()
                self.metrics[round_id] = metrics
                logging.info(
                    "Round Id : {} \n , Train -- Loss : {} , Acc : {}, \n Test-- Loss : {}, Acc : {} \n, Time :\n".format(
                        round_id,
                        metrics["train"]["loss"],
                        metrics["train"]["acc"],
                        metrics["test"]["loss"],
                        metrics["test"]["loss"],
                    )
                )
        self.model.eval()
        self.model.cpu()
        return metrics

    def compute_metrics(self, client_dict):
        metrics = []
        for i in self.client_idx:
            metrics.append(super().compute_metrics(client_dict[i]))
        return avg_metrics(metrics)

    def test(self, client_dict, client_idx):
        self.load_model_weights()
        self.client_idx = client_idx
        metrics = self.compute_metrics(client_dict)
        self.model.cpu()
        return metrics

    def send_to_clients(self, client_trainers, selected_clients):
        global_model = self.model.state_dict()
        for i in selected_clients:
            client_trainers[i].model.load_state_dict(global_model)
        return client_trainers

    def average_model(self, client_trainers, selected_clients):
        wts_list = [client_trainers[i].state_dict() for i in selected_clients]
        beta = self.config["beta"]
        ###### This needs a better name
        wts_dict_of_lists = {
            key: torch.stack([wt[key] for wt in wts_list], dim=0)
            for key in wts_list[0].keys()
        }

        start_idx = int(beta * len(selected_clients))
        end_idx = int((1 - beta) * len(selected_clients))
        if end_idx <= start_idx + 1:
            start_idx = 0
            end_idx = len(selected_clients)
            avg_wt = {key: item.mean(dim=0) for key, item in wts_dict_of_lists.items()}
        else:
            sorted_dict = {
                key: torch.sort(item, dim=0)[0]
                for key, item in wts_dict_of_lists.items()
            }
            avg_wt = {
                key: item[start_idx:end_idx, ...].mean(dim=0)
                for key, item in sorted_dict.items()
            }

        self.model.load_state_dict(avg_wt)
