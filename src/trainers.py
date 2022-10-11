import os
import random

from src.models.cnn import SimpleCNN
from src.models.linear import OneLayer, TwoLayer
from src.models.lstm import StackedLSTM
from src.models.resnet import ResNet9

MODEL_DICT = {
    "simplecnn": SimpleCNN,
    "one_layer_lin": OneLayer,
    "two_layer_lin": TwoLayer,
    "stacked_lstm": StackedLSTM,
    "resnet": ResNet9,
}

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from ray_lightning import RayStrategy
from tqdm import tqdm
from filelock import FileLock

from src.utils import (
    avg_metrics,
    check_nan,
    format_trainer_for_metrics,
    wt_dict_diff,
    wt_dict_norm,
    format_trainer_for_metrics,
)
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)


class BaseTrainer(pl.LightningModule):
    def __init__(self, config):
        super(BaseTrainer, self).__init__()
        self.config = config
        self.model = MODEL_DICT[self.config["model"]["name"]](self.config)
        self.lstm_flag = self.config["dataset"]["name"] == "shakespeare"
        self.save_dir = None

    def load_saved_weights(self):
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
            if self.config["tune"]:

                with FileLock(model_path + ".lock"):
                    torch.save(self.model.state_dict(), model_path)
            else:
                torch.save(self.model.state_dict(), model_path)

    def save_metrics(self):
        if self.metrics is not None:
            metrics_path = os.path.join(self.save_dir, "metrics.pth")
            if self.config["tune"]:
                with FileLock(metrics_path + ".lock"):
                    torch.save(self.metrics, metrics_path)
            else:
                torch.save(self.metrics, metrics_path)

    # def compute_loss(self, client_data, train=True):
    #     return compute_metric(
    #         self.model,
    #         client_data,
    #         train,
    #         loss=self.loss_func,
    #         device=self.device,
    #         lstm_flag=self.lstm_flag,
    #     )

    # def compute_acc(self, client_data, train=True):
    #     return compute_metric(
    #         self.model, client_data, train, device=self.device, lstm_flag=self.lstm_flag
    #     )

    # def compute_metrics(self, client_data):
    #     metrics = {"train": {}, "test": {}}
    #     # if self.config["dataset"]["name"] == "synthetic":
    #     #     assert self.device is not None
    #     # else:
    #     #     self.device = None
    #     self.check_model_on_device()

    #     metrics["train"]["loss"] = compute_metric(
    #         self.model,
    #         client_data,
    #         train=True,
    #         loss=self.loss_func,
    #         device=self.device,
    #         lstm_flag=self.lstm_flag,
    #     )
    #     metrics["test"]["loss"] = compute_metric(
    #         self.model,
    #         client_data,
    #         train=False,
    #         loss=self.loss_func,
    #         device=self.device,
    #         lstm_flag=self.lstm_flag,
    #     )
    #     if self.config["dataset"]["name"] not in "synthetic":
    #         metrics["train"]["acc"] = compute_metric(
    #             self.model,
    #             client_data,
    #             train=True,
    #             device=self.device,
    #             lstm_flag=self.lstm_flag,
    #         )
    #         metrics["test"]["acc"] = compute_metric(
    #             self.model,
    #             client_data,
    #             train=False,
    #             device=self.device,
    #             lstm_flag=self.lstm_flag,
    #         )
    #     return metrics

    # def test(self, client_data):
    #     self.load_model_weights()
    #     self.check_model_on_device()
    #     metrics = self.compute_metrics(client_data)
    #     self.model.cpu()
    #     return metrics

    # def check_model_on_device(self):
    #     if next(self.model.parameters()).device == torch.device("cpu"):
    #         self.model.to(self.device)

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

    def compute_metrics(self, client_data, trainer, train=False):
        # if self.config["tune"]:
        #     trainer = pl.Trainer(
        #         default_root_dir=self.save_dir,
        #         # progress_bar=TQDMProgressBar(refresh_rate=20),
        #         enable_model_summary=False,
        #         enable_progress_bar=False,
        #         strategy=RayStrategy(
        #             num_workers=3,
        #             num_cpus_per_worker=3,
        #             use_gpu=True,
        #         ),
        #         log_every_n_steps=1,
        #         precision=16,
        #         amp_backend="native",
        #         limit_train_batches=0,
        #         limit_val_batches=0,
        #     )
        # else:
        #     trainer = pl.Trainer(
        #         default_root_dir=self.save_dir,
        #         # progress_bar=TQDMProgressBar(refresh_rate=20),
        #         devices=torch.cuda.device_count(),
        #         accelerator="gpu",
        #         enable_model_summary=False,
        #         enable_progress_bar=False,
        #         strategy="ddp_find_unused_parameters_false",
        #         log_every_n_steps=1,
        #         precision=16,
        #         progress_bar_refresh_rate=10,
        #         amp_backend="native",
        #         limit_train_batches=0,
        #         limit_val_batches=0,
        #     )
        trainer = format_trainer_for_metrics(trainer, self.save_dir)
        trainer.test(
            self.model,
            client_data.train_dataloader() if train else client_data.test_dataloader(),
            verbose=False,
        )
        return trainer.logged_metrics


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

    def format_trainer_for_client(self, client_data, local_iter, trainer):
        trainer._default_root_dir = self.save_dir
        callbacks = []
        logger = None
        enable_progress_bar = False
        if self.config["dataset"]["name"] in ["synthetic", "shakespeare"]:
            metric_name = "val_loss"
            mode = "min"
        else:
            metric_name = "val_acc"
            mode = "max"
        # import ipdb

        # ipdb.set_trace()
        # if self.mode == "solo":
        #     early_stopping = EarlyStopping(
        #         monitor=metric_name,
        #         mode=mode,
        #         min_delta=1e-6,
        #         patience=5,
        #         verbose=False,
        #     )
        #     callbacks = [early_stopping]
        #     callbacks = []
        #     model_checkpoint = ModelCheckpoint(
        #         dirpath=self.save_dir,
        #         save_last=True,
        #         monitor=metric_name,
        #         mode=mode,
        #         every_n_train_steps=self.config["freq"]["save"],
        #     )
        #     callbacks.append(model_checkpoint)
        #     logger = CSVLogger(self.save_dir)
        #     progress_bar = TQDMProgressBar(refresh_rate=3)
        #     callbacks.append(progress_bar)
        #     enable_progress_bar = True

        # trainer.logger = logger
        # trainer.callbacks = callbacks
        import ipdb

        ipdb.set_trace()
        trainer.enable_progress_bar = enable_progress_bar
        trainer.fit_loop.max_steps = local_iter
        trainer.val_check_interval = max(
            min(
                local_iter,
                self.config["freq"]["metrics"],
                len(client_data.train_dataloader()),
            )
            - 1,
            1,
        )

        # trainer.check_val_every_n_epoch = 1
        # trainer.log_every_n_steps = max(
        #     min(local_iter, self.config["freq"]["metrics"]) - 1, 1
        # )
        # if trainer.accelerator is None:
        #     print("1")
        # trainer._logger_connector.on_trainer_init(
        #     logger,
        #     trainer.flush_logs_every_n_steps,
        #     trainer.log_every_n_steps,
        #     trainer.move_metrics_to_cpu,
        # )
        # if trainer.accelerator is None:
        #     import ipdb

        #     ipdb.set_trace()

        # trainer._callback_connector.on_trainer_init(
        #     callbacks=callbacks,
        #     checkpoint_callback=trainer.checkpoint_callback,
        #     enable_checkpointing=True,
        #     enable_progress_bar=True,
        #     progress_bar_refresh_rate=None,
        #     process_position=0,
        #     default_root_dir=trainer.default_root_dir,
        #     weights_save_path=trainer.weights_save_path,
        #     enable_model_summary=False,
        #     weights_summary=trainer.weights_summary,
        #     stochastic_weight_avg=False,
        #     max_time=None,
        #     accumulate_grad_batches=trainer.accumulate_grad_batches,
        # )
        # if trainer.accelerator is None:
        #     import ipdb

        #     ipdb.set_trace()

        # # hook
        # trainer._call_callback_hooks("on_init_start")
        # trainer._data_connector.on_trainer_init(
        #     trainer.check_val_every_n_epoch,
        #     trainer.reload_dataloaders_every_n_epochs,
        #     trainer.prepare_data_per_node,
        # )
        # if trainer.accelerator is None:
        #     import ipdb

        #     ipdb.set_trace()

        # trainer._init_debugging_flags(
        #     trainer.limit_train_batches,
        #     trainer.limit_val_batches,
        #     trainer.limit_test_batches,
        #     trainer.limit_predict_batches,
        #     trainer.val_check_interval,
        #     trainer.overfit_batches,
        #     trainer.fast_dev_run,
        # )

        # if trainer.accelerator is None:
        #     import ipdb

        #     ipdb.set_trace()
        # # Callback system
        # trainer._call_callback_hooks("on_init_end")

        return trainer

    def train(self, client_data, local_iter, trainer, rounds=None):
        # if self.mode == "solo":
        #     self.metrics = {}
        # else:
        #     metrics = None
        ## Set model wts
        # import ipdb

        # ipdb.set_trace()
        trainer = self.format_trainer_for_client(
            client_data=client_data, local_iter=local_iter, trainer=trainer
        )
        trainer.fit(
            self.model,
            client_data,
        )
        if self.mode == "solo":
            # trainer.test(self.model, client_data, verbose=False)
            self.metrics = trainer.logged_metrics
        else:
            self.metrics = None
        return self.metrics


class ClusterTrainer(BaseTrainer):
    def __init__(self, config, cluster_id, stop_threshold=None):
        super(ClusterTrainer, self).__init__(config)
        self.cluster_id = cluster_id
        self.stop_threshold = stop_threshold
        self.avg_wt = {}

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

    def train(self, client_dict, client_idx, local_iter, rounds, trainer):
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

        self.model.set_round_and_local_iter(local_iter, last_round - first_round)

        for round_id in tqdm(range(first_round, last_round + 1)):
            # metrics = []
            if self.config["num_clients_per_round"] <= len(client_idx):
                selected_clients = random.sample(
                    client_idx, self.config["num_clients_per_round"]
                )
            else:
                selected_clients = client_idx
            client_trainers = self.send_to_clients(client_trainers, selected_clients)
            for i in selected_clients:
                client_trainers[i].model.set_round_and_local_iter(local_iter, round_id)
                _ = client_trainers[i].train(
                    client_data=client_dict[i],
                    local_iter=local_iter,
                    rounds=round_id,
                    trainer=trainer,
                )
            self.average_model(client_trainers, selected_clients)

            if (round_id + 1) * local_iter % self.config["freq"][
                "metrics"
            ] == 0 or round_id == last_round - 1:
                metrics_list = []
                trainer = format_trainer_for_metrics(trainer, self.save_dir)
                for client_id in client_idx:
                    # if self.config["tune"]:
                    #     trainer = pl.Trainer(
                    #         default_root_dir=self.save_dir,
                    #         enable_model_summary=False,
                    #         max_steps=last_round - first_round,
                    #         enable_progress_bar=False,
                    #         strategy=RayStrategy(
                    #             num_workers=3,
                    #             num_cpus_per_worker=3,
                    #             use_gpu=True,
                    #         ),
                    #         log_every_n_steps=1,
                    #         logger=CSVLogger(self.save_dir),
                    #         precision=16,
                    #         enable_checkpointing=False,
                    #         amp_backend="native",
                    #         limit_train_batches=0,
                    #         limit_val_batches=0,
                    #     )

                    # else:
                    #     trainer = pl.Trainer(
                    #         # progress=TQDMProgressBar(refresh_rate=20),
                    #         default_root_dir=self.save_dir,
                    #         devices=torch.cuda.device_count(),
                    #         accelerator="gpu",
                    #         enable_model_summary=False,
                    #         max_steps=last_round - first_round,
                    #         enable_progress_bar=False,
                    #         strategy="ddp_find_unused_parameters_false",
                    #         log_every_n_steps=1,
                    #         logger=CSVLogger(self.save_dir),
                    #         precision=16,
                    #         enable_checkpointing=False,
                    #         amp_backend="native",
                    #         limit_train_batches=0,
                    #         limit_val_batches=0,
                    #     )

                    trainer.test(
                        self.model,
                        client_dict[client_id],
                        verbose=False,
                        ckpt_path=None,
                    )
                    metrics_list.append((1, trainer.logged_metrics))
                self.metrics[round_id] = avg_metrics(metrics_list)
                if check_nan(self.metrics[round_id]):
                    # return self.metrics[round_id]
                    raise ValueError("Nan or inf occurred in metrics")

            #     metrics = self.compute_metrics(client_dict)
            #     self.metrics[round_id] = metrics
            #     if check_nan(metrics):
            #         self.model.eval()
            #         self.model.cpu()
            #         raise ValueError("Nan or inf occurred in metrics")
            #         # return metrics
            if self.stop_threshold is not None:
                self.client_wt_diff = {
                    i: wt_dict_diff(
                        client_trainers[i].get_model_wts(), self.prev_model_wt
                    )
                    for i in self.client_idx
                }
                if self.stop_at_threshold():
                    if round_id in self.metrics.keys():
                        return self.metrics[round_id]
                    else:
                        return None
                    # self.model.eval()
                    # self.model.cpu()
                    # return metrics
            else:
                self.client_wt_diff = {}

            if (round_id + 1) % self.config["freq"][
                "save"
            ] == 0 or round_id == last_round - 1:
                self.save_model_weights()
                self.save_metrics()
            if (round_id + 1) % self.config["freq"][
                "print"
            ] == 0 or round_id == last_round - 1:
                print(
                    "Round Id : {} \n , Metrics : {}\n".format(
                        round_id + 1, self.metrics[round_id]
                    )
                )
        if last_round - 1 in self.metrics.keys():

            return self.metrics[last_round - 1]
        else:
            return None

    def compute_metrics(self, client_dict, train=False):
        metrics = []
        for i in self.client_idx:
            metrics.append((1, super().compute_metrics(client_dict[i], train=train)))
        return avg_metrics(metrics)

    # def test(self, client_dict, client_idx):
    #     self.load_saved_weights()
    #     self.client_idx = client_idx
    #     metrics = self.compute_metrics(client_dict)
    #     self.model.cpu()
    #     return metrics

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
            key: torch.stack([wt[key].to("cpu") for wt in wts_list], dim=0)
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
