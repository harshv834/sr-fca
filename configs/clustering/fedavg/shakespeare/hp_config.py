from math import ceil

from ray import tune

# class Config:
#     def __init__(self):
#         self.config = {
#             "model": {"name": "simplelin"},
#         }

model: 
  name: stacked_lstm
  params:
    seq_len : 80
    emb_dim: 8
    n_hidden: 100
    num_classes : 80
    n_layers: 2
freq:
  metrics: 5
  save: 150
  print: 100
num_clients_per_round: 3
local_iter: 5
rounds: 2
optimizer:
  name: sgd
  params:
    lr: 0.003
    momentum: 0.9

def get_hp_config(trial, data_config):
    local_iter = trial.suggest_int("local_iter", 1, 10)
    optimizer_name = trial.suggest_categorical("optimizer_name", ["sgd", "adam"])
    if optimizer_name == "sgd":
        lr = trial.suggest_loguniform("optimizer_params_lr", 5e-4, 1e-2)
        optimizer_param_dict = {"lr": lr, "momentum": momentum}
    else:
        lr = trial.suggest_loguniform("optimizer_params_lr", 1e-4, 1e-2)
        optimizer_param_dict = {"lr": lr}

    config = {
        "model": {
            "name": "stacked_lstm",
            "params": {
                "seq_len" : data_config["seq_len"]
            },
        },
        "dist_threshold": dist_threshold,
        "size_threshold": size_threshold,
        "freq": {"metrics": 5, "save": 150, "print": 100},
        "beta": beta,
        "init": {"iterations": 300},
        "refine": {"local_iter": refine_local_iter},
        "num_refine_steps": num_refine_steps,
        "optimizer": {"name": optimizer_name, "params": optimizer_param_dict},
    }
    config = data_config | config

    return config
