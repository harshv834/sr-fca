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
        momentum = trial.suggest_uniform("optimizer_params_momentum", 0.1, 0.9)
        optimizer_param_dict = {"lr": lr, "betas": (momentum, 0.999)}

    config = {
        "model": {
            "name": "simplelin",
            "params": {
                "dimension": data_config["dataset"]["dimension"],
                "scale": data_config["dataset"]["scale"],
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
