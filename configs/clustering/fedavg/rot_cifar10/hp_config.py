from math import ceil

from ray import tune

# class Config:
#     def __init__(self):
#         self.config = {
#             "model": {"name": "simplelin"},
#         }
def get_hp_config(trial, data_config):
    # trial.suggest_categorical("model", [{"name": "simplelin"}])
    # trial.suggest_categorical("freq",[{"metrics": 5, "save": 150, "print": 100}])
    local_iter = trial.suggest_int("local_iter", 1, 5)
    optimizer_name = trial.suggest_categorical("optimizer_name", ["sgd", "adam"])
    if optimizer_name == "sgd":
        lr = trial.suggest_loguniform("optimizer_params_lr", 1e-3, 1e-1)
        momentum = trial.suggest_uniform("optimizer_params_momentum", 0.0, 0.9)
        optimizer_param_dict = {"lr": lr, "momentum": momentum}
    else:
        lr = trial.suggest_loguniform("optimizer_params_lr", 1e-4, 1e-2)
        optimizer_param_dict = {"lr": lr}
    num_clients_per_round = trial.suggest_int("num_clients_per_round", 2, 10)
    config = {
        "model": {
            "name": "resnet",
            "params": {
                "num_classes": data_config["dataset"]["num_classes"],
            },
        },
        "freq": {"metrics": 30, "save": 60, "print": 30},
        "iterations": 20,
        "local_iter": local_iter,
        "num_clients_per_round": num_clients_per_round,
        "optimizer": {"name": optimizer_name, "params": optimizer_param_dict},
        "label_smoothing": 0.1,
    }
    config = data_config | config
    return config
