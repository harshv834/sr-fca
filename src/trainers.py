import os
import random
from abc import ABC
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
from src.models.base import MODEL_DICT
from src.utils import (
    avg_metrics,
    check_nan,
    compute_metric,
    get_device,
    get_lr_scheduler,
    get_optimizer,
    set_weights,
    wt_dict_diff,
    wt_dict_norm,
)
from functools import partialmethod


from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights, resnet18

# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


class BaseTrainer(ABC):
    def __init__(self, config):
        super(BaseTrainer, self).__init__()
        self.config = config
        if self.config["dataset"]["name"] == "rot_cifar10_ftrs":
            self.model = resnet18(pretrained=True)
            # self.model = efficientnet_b7(EfficientNet_B7_Weights)
            for param in self.model.parameters():
                param.requires_grad = False
            num_classes = self.config["dataset"]["num_classes"]
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            # children = list(self.model.children())
            # self.model = nn.Sequential(*children[:-1],
            #               nn.Sequential(nn.Dropout(p=0.5, inplace=True),
            #                             nn.Linear(list(children[-1].children())[-1].in_features, num_classes)))

        else:
            model = MODEL_DICT[self.config["model"]["name"]]
            if "params" in self.config["model"].keys():
                self.model = model(**self.config["model"]["params"])
            else:
                self.model = model()
        self.lstm_flag = self.config["dataset"]["name"] == "shakespeare"
        if self.config["dataset"]["name"] == "synthetic":
            self.loss_func = nn.MSELoss()
        else:
            label_smoothing = (
                self.config["label_smoothing"]
                if "label_smoothing" in self.config.keys()
                else 0.0
            )
            self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # if self.config["dataset"]["name"] in ["femnist"]:

        #     # if self.config["dataset"]["name"] in ["femnist", "rot_cifar10", "shakespeare"]:
        #     self.model = set_weights(
        #         self.config["dataset"]["name"],
        #         self.model,
        #         self.config["pretrained_path"],
        #     )

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def load_saved_weights(self):
        """Load saved model weights

        Raises:
            ValueError: when no saved weights present.
        """
        model_path = os.path.join(self.save_dir, "model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
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
            self.model,
            client_data,
            train,
            loss=self.loss_func,
            device=self.device,
            lstm_flag=self.lstm_flag,
        )

    def compute_acc(self, client_data, train=True):
        return compute_metric(
            self.model, client_data, train, device=self.device, lstm_flag=self.lstm_flag
        )

    def compute_loss_list(self, client_data, train=True):
        if self.config["dataset"]["name"] == "synthetic":
            self.loss_func = nn.MSELoss(reduction=None)
        else:
            label_smoothing = (
                self.config["label_smoothing"]
                if "label_smoothing" in self.config.keys()
                else 0.0
            )
            loss_fn_no_reduce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')
        return compute_metric(self.model, client_data, train,loss=loss_fn_no_reduce, device=self.device, lstm_flag=self.lstm_flag, return_list= True)
    

    def compute_metrics(self, client_data):
        metrics = {"train": {}, "test": {}}
        # if self.config["dataset"]["name"] == "synthetic":
        #     assert self.device is not None
        # else:
        #     self.device = None
        self.check_model_on_device()

        metrics["train"]["loss"] = compute_metric(
            self.model,
            client_data,
            train=True,
            loss=self.loss_func,
            device=self.device,
            lstm_flag=self.lstm_flag,
        )
        metrics["test"]["loss"] = compute_metric(
            self.model,
            client_data,
            train=False,
            loss=self.loss_func,
            device=self.device,
            lstm_flag=self.lstm_flag,
        )
        if self.config["dataset"]["name"] not in "synthetic":
            metrics["train"]["acc"] = compute_metric(
                self.model,
                client_data,
                train=True,
                device=self.device,
                lstm_flag=self.lstm_flag,
            )
            metrics["test"]["acc"] = compute_metric(
                self.model,
                client_data,
                train=False,
                device=self.device,
                lstm_flag=self.lstm_flag,
            )
        return metrics

    def test(self, client_data):
        self.load_model_weights()
        self.check_model_on_device()
        metrics = self.compute_metrics(client_data)
        self.model.cpu()
        return metrics

    def check_model_on_device(self):
        if self.config["dataset"]["name"].startswith("rot_cifar10"):
            self.model = self.model.cuda()
        else:
            if next(self.model.parameters()).device != torch.device(self.device):
                self.model = self.model.to(self.device)

    def get_model_wts(self):
        wts = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                wts[name] = param.data
        return wts

    def load_wts_from_dict(self, wts):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = wts[name]


class ClientTrainer(BaseTrainer):
    def __init__(self, config, client_id, mode="fed"):
        super(ClientTrainer, self).__init__(config)
        self.client_id = client_id
        self.mode = mode
        if self.mode not in ["fed", "solo"]:
            raise ValueError("Invalid mode for ClientTrainer {}".format(self.mode))
        self.device = get_device(self.config, self.client_id)

    def set_save_dir(self, save_dir):
        if self.mode == "solo":
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)

    def train(self, client_data, local_iter, round_id=None):
        if self.mode == "solo":
            self.metrics = {}
        else:
            metrics = None
        optimizer = get_optimizer(self.model.parameters(), self.config)
        if self.config["dataset"]["name"] in [
            "rot_cifar10",
            "shakespeare",
            "rot_cifar10_ftrs",
        ]:
            scheduler = get_lr_scheduler(self.config, optimizer, local_iter, round_id)
            scaler = GradScaler()
        if self.config["dataset"]["name"].startswith("rot_cifar10"):
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.lstm_flag:
            batch_size, hidden = None, None
        # flag = True
        # import ipdb

        # ipdb.set_trace()
        for iteration in tqdm(range(local_iter), disable=self.mode == "fed"):
            t0 = time.time()
            # if self.config["dataset"]["name"] == "rot_cifar10_ftrs":
            #     if self.mode == "solo" and flag:
            #         if local_iter >= 500:
            #             for param in self.model.parameters():
            #                 param.requires_grad=True
            #         flag=False
            self.model.train()
            self.check_model_on_device()

            optimizer.zero_grad(set_to_none=True)
            (X, Y) = client_data.sample_batch(train=True)
            if self.lstm_flag:
                if batch_size is None or X.shape[0] != batch_size:
                    batch_size = X.shape[0]
                    hidden = self.model.zero_state(batch_size)
            if not self.config["dataset"]["name"].startswith("rot_cifar10"):
                X, Y = X.to(self.device), Y.to(self.device)
            if self.lstm_flag:
                hidden = (hidden[0].to(self.device), hidden[1].to(self.device))

            if self.config["dataset"]["name"] in [
                "rot_cifar10",
                "shakespeare",
                "rot_cifar10",
            ]:
                with autocast():
                    if self.lstm_flag:
                        out, hidden = self.model(X, hidden)
                        hidden = (hidden[0].detach(), hidden[1].detach())
                    else:
                        out = self.model(X)

                    loss = self.loss_func(out, Y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    # scaler.unscale_(optimizer)
                    # loss.backward()
                    # optimizer.step()
                    scaler.update()
                    scheduler.step()

            else:
                out = self.model(X)
                loss = self.loss_func(out, Y)
                loss.backward()
                optimizer.step()

            t1 = time.time()
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
                        "Iteration : {} \n , Metrics : {}, Time : {}\n".format(
                            iteration, metrics, t1 - t0
                        )
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
            model_wt = self.get_model_wts()
            diff_wt = wt_dict_diff(model_wt, self.prev_model_wt)
            return wt_dict_norm(diff_wt) < self.stop_threshold

    def train(self, client_dict, client_idx, local_iter, rounds, client_sampling_wts=None):
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

        # import ipdb;ipdb.set_trace()
        if self.config["dataset"]["name"] == "rot_cifar10":
            self.model = self.model.to(memory_format=torch.channels_last)
        for round_id in tqdm(
            range(first_round, last_round + 1),
            disable=self.config["clustering"] == "ifca",
        ):
            # if self.config["dataset"]["name"] == "rot_cifar10_ftrs":
            #     if round_id * local_iter >= 500 and flag:
            #         for param in self.model.parameters():
            #             param.requires_grad=True
            #         flag = False
            t0 = time.time()
            self.model.train()
            self.check_model_on_device()
            metrics = []
            if self.config["num_clients_per_round"] <= len(client_idx):
                if client_sampling_wts is None:
                    client_sampling_wts = np.ones(len(client_idx))/len(client_idx)
                #selected_clients = random.sample(
                #    client_idx, self.config["num_clients_per_round"], 
                #)
                selected_clients =  np.random.choice(
                        client_idx, 
                        size=self.config["num_clients_per_round"], 
                        replace=False, 
                        p=client_sampling_wts
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
                        client_trainers[i].get_model_wts(), self.prev_model_wt
                    )
                    for i in self.client_idx
                }
                if self.stop_at_threshold():
                    self.model.eval()
                    self.model.cpu()
                    return metrics
            else:
                self.client_wt_diff = {}
            t1 = time.time()
            if (
                round_id % self.config["freq"]["save"] == 0
                or round_id == last_round
            ):
                self.save_model_weights()
                self.save_metrics()
            if (
                round_id % self.config["freq"]["print"] == 0
                or round_id == last_round
            ):
                print(
                    "Round Id : {} \n , Metrics : {}, Time taken : {}\n".format(
                        round_id, metrics, t1 - t0
                    )
                )
        self.model.eval()
        self.model.cpu()
        return metrics
    def compute_loss_list(self, client_dict, train=True):
        client_loss_list  = []
        for i in self.client_idx:
            self.check_model_on_device()
            client_loss_list.append(super().compute_loss_list(client_dict[i], train=train))
        return client_loss_list

    def compute_metrics(self, client_dict):
        metrics = []
        for i in self.client_idx:
            self.check_model_on_device()
            metrics.append((1, super().compute_metrics(client_dict[i])))
        return avg_metrics(metrics)

    def test(self, client_dict, client_idx):
        self.load_saved_weights()
        self.client_idx = client_idx
        metrics = self.compute_metrics(client_dict)
        self.model.cpu()
        return metrics

    def send_to_clients(self, client_trainers, selected_clients):
        global_model = self.get_model_wts()
        for i in selected_clients:
            client_trainers[i].load_wts_from_dict(global_model)
        return client_trainers

    def average_model(self, client_trainers, selected_clients):
        wts_list = [client_trainers[i].get_model_wts() for i in selected_clients]
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
        self.prev_model_wt = self.get_model_wts()
        self.load_wts_from_dict(avg_wt)
