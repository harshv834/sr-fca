from math import ceil

from ray import tune

# class Config:
#     def __init__(self):
#         self.config = {
#             "model": {"name": "simplelin"},
#         }


def get_hp_config(trial, data_config):
    # trial.suggest_categorical("model", [{"name": "simplelin"}])
    dist_threshold = (trial.suggest_loguniform("dist_threshold", 10, 100),)
    size_threshold = (trial.suggest_int("size_threshold", 1, 10),)
    # trial.suggest_categorical("freq",[{"metrics": 5, "save": 150, "print": 100}])
    beta = trial.suggest_uniform("beta", 0.1, 0.4)
    # trial.suggest_categorical("init", [{"iterations": 300}])
    num_refine_steps = trial.suggest_int("num_refine_steps", 1, 4)
    refine_local_iter = trial.suggest_int("refine_local_iter", 1, 5)
    optimizer_name = trial.suggest_categorical("optimizer_name", ["sgd", "adam"])
    if optimizer_name == "sgd":
        lr = trial.suggest_loguniform("optimizer_params_lr", 1e-3, 1e-1)
        optimizer_param_dict = {"lr": lr}
    else:
        lr = trial.suggest_loguniform("optimizer_params_lr", 1e-4, 1e-2)
        optimizer_param_dict = {"lr": lr}
    num_clients_per_round = trial.suggest_int("num_clients_per_round", 1,10)

    config = {
        "model": {
            "name": "two_layer_lin",
            "params": {
                "input_size": data_config["dataset"]["input_size"],
                "num_classes": data_config["dataset"]["num_classes"],
                "hidden_size": data_config["dataset"]["hidden_size"]
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
        "num_clients_per_round": 
    }
    config = data_config | config

    return config
