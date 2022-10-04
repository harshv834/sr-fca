import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
import torchmetrics
import torch
import numpy as np


class LossMetric(torchmetrics.Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    correct: torch.Tensor
    total: torch.Tensor

    def __init__(self, loss_func):
        super(LossMetric, self).__init__()
        self.loss_func = loss_func
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        loss = self.loss_func(preds, target)
        self.correct += loss.item()
        self.total += target.shape[0]

    def compute(self):
        return self.correct.float() / self.total


class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        if self.config["dataset"]["name"] == "synthetic":
            self.loss_func = nn.MSELoss()
        else:
            label_smoothing = (
                self.config["label_smoothing"]
                if "label_smoothing" in self.config.keys()
                else 0.0
            )
            self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.train_loss = LossMetric(self.loss_func)
        self.val_loss = LossMetric(self.loss_func)
        self.test_loss = LossMetric(self.loss_func)
        if self.config["dataset"]["name"] != "synthetic":
            self.test_acc = torchmetrics.Accuracy()
            self.train_acc = torchmetrics.Accuracy()
            self.val_acc = torchmetrics.Accuracy()
        self.local_iter = None
        self.rounds = None

    def set_round_and_local_iter(self, local_iter, rounds):
        self.local_iter = local_iter
        self.rounds = rounds

    def training_step(self, batch, batch_idx):
        (X, Y) = batch
        out = self.forward(X)
        loss = self.loss_func(out, Y)

        self.train_loss.update(out, Y)
        self.log("train_loss", self.train_loss, on_epoch=True, prog_bar=True)

        if self.config["dataset"]["name"] != "synthetic":
            self.train_acc.update(out, Y)
            self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer_name = self.config["optimizer"]["name"]
        optimizer_params = self.config["optimizer"]["params"]
        if optimizer_name == "sgd":
            optimizer = optim.SGD(self.parameters(), **optimizer_params)
        elif optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), **optimizer_params)
        else:
            raise ValueError("Invalid optimizer name {}".format(optimizer_name))

        lr_scheduler = self.get_lr_scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

    def get_lr_scheduler(self, optimizer):
        cond_1 = self.config["dataset"]["name"] == "rot_cifar10"
        cond_2 = self.config["clustering"] == "sr_fca"
        if cond_1 and cond_2:
            iters_per_epoch = 50000 // int(
                self.config["batch"]["train"]
                * (self.config["num_clients"] // self.config["dataset"]["num_clusters"])
            )
            epochs = self.config["init"]["iterations"] // iters_per_epoch
            lr_schedule = np.interp(
                np.arange((epochs + 1) * iters_per_epoch),
                [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
                [0, 1, 0],
            )
            if self.rounds is not None:
                if type(self.rounds) == tuple:
                    first_iter = self.rounds[0] * self.local_iter
                    last_iter = max(
                        (self.rounds[1] + 1) * self.local_iter, lr_schedule.shape[0]
                    )
                else:
                    first_iter = 0
                    last_iter = max(
                        (self.rounds + 1) * self.local_iter, lr_schedule.shape[0]
                    )
                lr_schedule = lr_schedule[first_iter:last_iter]
        elif not cond_1 and cond_2:
            lr_schedule = np.ones(self.config["init"]["iterations"] + 1)
        else:
            lr_schedule = np.ones(max(self.config["rounds"], self.local_iter) + 1)

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=lambda i: lr_schedule[i]
        )
        return scheduler

    def test_step(self, batch, batch_idx):
        (X, Y) = batch
        out = self.forward(X)

        self.test_loss.update(out, Y)
        self.log("test_loss", self.test_loss, prog_bar=True)
        if self.config["dataset"]["name"] != "synthetic":
            self.test_acc.update(out, Y)
            self.log("test_acc", self.test_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        (X, Y) = batch
        out = self.forward(X)
        self.val_loss.update(out, Y)
        self.log("val_loss", self.val_loss, prog_bar=True)

        if self.config["dataset"]["name"] != "synthetic":
            self.val_acc.update(out, Y)
            self.log("val_acc", self.val_acc, prog_bar=True)
