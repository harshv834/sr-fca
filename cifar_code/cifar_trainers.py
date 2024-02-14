from abc import ABC
import os
import torch
import numpy as np
import time
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from model import ResNet9
from cifar_utils import calc_acc
from tqdm import tqdm
from cifar_utils import calc_acc

MODEL_LIST = {"resnet9": ResNet9}
OPTIMIZER_LIST = {"sgd": optim.SGD, "adam": optim.Adam}
LOSSES = {"cross_entropy": nn.CrossEntropyLoss(label_smoothing=0.1)}



class BaseTrainer(ABC):
    def __init__(self, config, save_dir):
        super(BaseTrainer, self).__init__()
        self.model = MODEL_LIST[config["model"]](**config["model_params"])
        self.save_dir = save_dir
        self.device = config["device"]
        self.loss_func = LOSSES[config["loss_func"]]
        self.config = config
        os.makedirs(self.save_dir, exist_ok=True)

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
            print("No model present at path : {}".format())

    def save_model_weights(self):
        model_path = os.path.join(self.save_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)

    def save_metrics(self, train_loss, test_acc, iteration):
        torch.save(
            {"train_loss": train_loss, "test_acc": test_acc},
            os.path.join(self.save_dir, "metrics_{}.pkl".format(iteration)),
        )



class ClientTrainer(BaseTrainer):
    def __init__(self, config, save_dir, client_id):
        super(ClientTrainer, self).__init__(config, save_dir)
        self.client_id = client_id

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, client_data):
        train_loss_list = []
        test_acc_list = []
        self.model.to(memory_format=torch.channels_last).cuda()
        # (self.device)
        self.model.train()
        optimizer = OPTIMIZER_LIST[self.config["optimizer"]](
            self.model.parameters(), **self.config["optimizer_params"]
        )
        iters_per_epoch = 50000 // int(
            self.config["train_batch"] * self.config["total_num_clients_per_cluster"]
        )
        epochs = self.config["iterations"] // iters_per_epoch
        lr_schedule = np.interp(
            np.arange((epochs + 1) * iters_per_epoch),
            [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
            [0, 1, 0],
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
        scaler = GradScaler()

        for iteration in tqdm(range(self.config["iterations"])):
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            (X, Y) = client_data.sample_batch(train=True)
            with autocast():
                out = self.model(X)
                loss = self.loss_func(out, Y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss = loss.detach().cpu().numpy().item()
            train_loss_list.append(train_loss)
            test_acc = calc_acc(self.model, self.device, client_data, train=False)
            test_acc_list.append(test_acc)
            self.model.train()
            t1 = time.time()
            time_taken = t1 - t0
            if (
                iteration % self.config["save_freq"] == 0
                or iteration == self.config["iterations"] - 1
            ):
                self.save_model_weights()
                self.save_metrics(train_loss_list, test_acc_list, iteration)

            if (
                iteration % self.config["print_freq"] == 0
                or iteration == self.config["iterations"] - 1
            ):
                print(
                    "Iteration : {} \n , Train Loss : {} \n, Test Acc : {} \n, Time : {}\n".format(
                        iteration, train_loss, test_acc, time_taken
                    )
                )

        self.model.eval()
        self.model.cpu()

    def test(self, client_data):
        self.load_model_weights()
        self.model.eval()
        self.model.to(self.device)
        acc = calc_acc(self.model, client_data)
        self.model.cpu()
        return acc



class ClusterTrainer(BaseTrainer):
    def __init__(self, config, save_dir, cluster_id):
        super(ClusterTrainer, self).__init__(config, save_dir)
        self.cluster_id = cluster_id

    def train(self, client_data_list):
        num_clients = len(client_data_list)

        train_loss_list = []
        test_acc_list = []
        self.model.to(memory_format=torch.channels_last).cuda()
        self.model.train()
        optimizer = OPTIMIZER_LIST[self.config["optimizer"]](
            self.model.parameters(), **self.config["optimizer_params"]
        )
        # eff_num_workers = int(num_clients/(1 - 2*beta))
        # if eff_num_workers > 0:
        #     eff_batch_size = self.config["train_batch"]/eff_num_workers
        #     for i in range(num_clients):
        #         client_data_list[i].trainloader.batch_size = eff_batch_size

        iters_per_epoch = 50000 * 0.5 // int(
            self.config["train_batch"] * self.config["total_num_clients_per_cluster"]
        )
        epochs = self.config["iterations"] // iters_per_epoch
        lr_schedule = np.interp(
            np.arange((epochs + 1) * iters_per_epoch),
            [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
            [0, 1, 0],
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
        scaler = GradScaler()

        for iteration in tqdm(range(self.config["iterations"])):
            t0 = time.time()
            trmean_buffer = {}
            for idx, param in self.model.named_parameters():
                trmean_buffer[idx] = []
            train_loss = 0
            for client in client_data_list:
                optimizer.zero_grad(set_to_none=True)
                (X, Y) = client.sample_batch()
                loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
                with autocast():
                    out = self.model(X)
                    loss = loss_func(out, Y)
                scaler.scale(loss).backward()
                train_loss += loss.detach().cpu().numpy().item()
                with torch.no_grad():
                    for idx, param in self.model.named_parameters():
                        trmean_buffer[idx].append(param.grad.clone())

            optimizer.zero_grad(set_to_none=True)
            start_idx = int(self.config["beta"] * num_clients)
            end_idx = int((1 - self.config["beta"]) * num_clients)
            if end_idx <= start_idx + 1:
                start_idx = 0
                end_idx = num_clients

            for idx, param in self.model.named_parameters():
                sorted, _ = torch.sort(torch.stack(trmean_buffer[idx], dim=0), dim=0)
                new_grad = sorted[start_idx:end_idx, ...].mean(dim=0)
                param.grad = new_grad
                trmean_buffer[idx] = []
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss = train_loss / num_clients
            train_loss_list.append(train_loss)
            test_acc = 0
            for client_data in client_data_list:
                test_acc += calc_acc(self.model, self.device, client_data, train=False)
            test_acc = test_acc / num_clients
            test_acc_list.append(test_acc)
            self.model.train()
            t1 = time.time()
            time_taken = t1 - t0

            if (
                iteration % self.config["save_freq"] == 0
                or iteration == self.config["iterations"] - 1
            ):
                self.save_model_weights()
                self.save_metrics(train_loss_list, test_acc_list, iteration)

            if iteration % self.config["print_freq"] == 0 or iteration  == self.config["iterations"] - 1:
                print(
                    "Iteration : {} \n , Train Loss : {} \n, Test Acc : {} \n, Time : {}\n".format(
                        iteration, train_loss, test_acc, time_taken
                    )
                )

        self.model.eval()
        # self.model.cpu()

    def test(self, client_data_list):
        self.load_model_weights()
        self.model.eval()
        self.model.to(self.device)
        test_acc = 0
        for client_data in client_data_list:
            test_acc += calc_acc(self.model, self.device, client_data, train=False)
        self.model.cpu()
        return test_acc
