from src.models.base import MODEL_DICT
from abc import ABC
import torch.nn as nn
import os
import torch
from torch.cuda.amp import autocast, GradScaler
from src.utils import (
    compute_metric,
    get_optimizer,
    get_lr_scheduler,
    avg_metrics,
    get_device,
    check_nan,
    wt_dict_diff,
    wt_dict_norm,
)
from tqdm import tqdm
import random


class BaseTrainer(ABC):
    def __init__(self, config):
        super(BaseTrainer, self).__init__()
        self.config = config
        self.model = MODEL_DICT[self.config["model"]["name"]](
            **self.config["model"]["params"]
        )
        if self.config["dataset"]["name"] == "synthetic":
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
        return compute_metric(
            self.model, client_data, train, loss=self.loss_func, device=self.device
        )

    def compute_acc(self, client_data, train=True):
        return compute_metric(self.model, client_data, train, device=self.device)

    def compute_metrics(self, client_data):
        metrics = {"train": {}, "test": {}}
        if self.config["dataset"]["name"] == "synthetic":
            assert self.device is not None
        else:
            self.device = None

        metrics["train"]["loss"] = compute_metric(
            self.model,
            client_data,
            train=True,
            loss=self.loss_func,
            device=self.device,
        )
        metrics["test"]["loss"] = compute_metric(
            self.model,
            client_data,
            train=False,
            loss=self.loss_func,
            device=self.device,
        )
        if self.config["dataset"]["name"] != "synthetic":
            metrics["train"]["acc"] = compute_metric(
                self.model, client_data, train=True, device=self.device
            )
            metrics["test"]["acc"] = compute_metric(
                self.model, client_data, train=False, device=self.device
            )
        return metrics

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
        if self.config["dataset"]["name"] == "synthetic":
            self.device = torch.device(get_device(self.config, self.client_id))
        else:
            self.device = None

    def set_save_dir(self, save_dir):
        if self.mode == "solo":
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)

    def train(self, client_data, local_iter, round=None):
        if self.mode == "solo":
            self.metrics = {}
        else:
            metrics = None

        optimizer = get_optimizer(self.model.parameters(), self.config)
        scheduler = get_lr_scheduler(self.config, optimizer, local_iter, round)
        scaler = GradScaler()

        for iteration in tqdm(range(local_iter)):
            self.model.train()
            self.model = self.model.to(memory_format=torch.channels_last).cuda()
            optimizer.zero_grad(set_to_none=True)
            (X, Y) = client_data.sample_batch(train=True)

            if self.config["dataset"]["name"] == "synthetic":
                X, Y = X.to(self.device), Y.to(self.device)
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
                    iteration % self.config["freq"]["metrics"] == 0
                    or iteration == local_iter - 1
                ):
                    self.model.eval()
                    metrics = self.compute_metrics(client_data)
                    self.metrics[iteration] = metrics
                    if check_nan(metrics):
                        self.model.eval()
                        self.model.cpu()
                        raise ValueError("Nan or inf occurred in metrics")
                        # return metrics
                if (
                    iteration % self.config["freq"]["save"] == 0
                    or iteration == local_iter - 1
                ):
                    self.save_model_weights()
                    self.save_metrics()
                if (
                    iteration % self.config["freq"]["print"] == 0
                    or iteration == local_iter - 1
                ):
                    print(
                        "Iteration : {} \n , Metrics : {}\n".format(iteration, metrics)
                    )

        self.model.eval()
        self.model.cpu()
        return metrics


class ClusterTrainer(BaseTrainer):
    def __init__(self, config, cluster_id, stop_threshold=None):
        super(ClusterTrainer, self).__init__(config)
        self.cluster_id = cluster_id
        self.stop_threshold = stop_threshold
        self.avg_wt = {}
        self.device = get_device(config, cluster_id, cluster=True)

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def stop_at_threshold(self):
        if self.stop_threshold is None:
            return False
        else:
            model_wt = self.model.state_dict()
            diff_wt = wt_dict_diff(model_wt, self.prev_model_wt)
            return wt_dict_norm(diff_wt) < self.stop_threshold

    def train(self, client_dict, client_idx, local_iter, rounds):
        self.num_clients = len(client_idx)
        self.client_idx = client_idx
        client_trainers = {
            i: ClientTrainer(config=self.config, client_id=i, mode="fed")
            for i in client_idx
        }
        if not hasattr(self, "metrics"):
            self.metrics = {}

        if type(rounds) == tuple:
            last_round = rounds[-1] - 1
            first_round = rounds[0]
        else:
            last_round = rounds - 1
            first_round = 0
        import ipdb

        ipdb.set_trace()
        for round_id in tqdm(range(first_round, last_round + 1)):
            self.model.train()
            self.model = self.model.to(memory_format=torch.channels_last).cuda()
            metrics = []
            if (
                self.config["num_clients_per_round"] <= len(client_idx)
                and self.config["clustering"] == "ifca"
            ):
                selected_clients = random.sample(
                    client_idx, self.config["num_clients_per_round"]
                )
            else:
                selected_clients = client_idx
            client_trainers = self.send_to_clients(client_trainers, selected_clients)
            for i in selected_clients:
                _ = client_trainers[i].train(client_dict[i], local_iter, round_id)
            self.average_model(client_trainers, selected_clients)
            if (
                round_id * local_iter % self.config["freq"]["metrics"] == 0
                or round_id == last_round
            ):
                self.model.eval()
                metrics = self.compute_metrics(client_dict)
                self.metrics[round_id] = metrics
                if check_nan(metrics):
                    self.model.eval()
                    self.model.cpu()
                    raise ValueError("Nan or inf occurred in metrics")
                    # return metrics
            if self.stop_threshold is not None:
                self.client_wt_diff = {
                    i: wt_dict_diff(
                        client_trainers[i].model.state_dict(), self.prev_model_wt
                    )
                    for i in self.client_idx
                }
                if self.stop_at_threshold():
                    self.model.eval()
                    self.model.cpu()
                    return metrics
            else:
                self.client_wt_diff = {}
            if (
                round_id % self.config["freq"]["save"] == 0
                or round_id == last_round - 1
            ):
                self.save_model_weights()
                self.save_metrics()
            if (
                round_id % self.config["freq"]["print"] == 0
                or round_id == last_round - 1
            ):
                print("Round Id : {} \n , Metrics : {}\n".format(round_id, metrics))
        self.model.eval()
        self.model.cpu()
        return metrics

    def compute_metrics(self, client_dict):
        metrics = []
        for i in self.client_idx:
            metrics.append((1, super().compute_metrics(client_dict[i])))
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
        wts_list = [client_trainers[i].model.state_dict() for i in selected_clients]
        if "beta" in self.config.keys():
            beta = self.config["beta"]
        else:
            beta = 0

        ###### This needs a better name
        wts_dict_of_lists = {
            key: torch.stack([wt[key] for wt in wts_list], dim=0)
            for key in wts_list[0].keys()
        }

        start_idx = int(beta * len(selected_clients))
        end_idx = int((1 - beta) * len(selected_clients))
        if end_idx <= start_idx + 1 or beta == 0:
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
        self.prev_model_wt = self.model.state_dict()
        self.model.load_state_dict(avg_wt)
