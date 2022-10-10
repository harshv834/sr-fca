def get_hp_config(trial, data_config):
    dist_fraction = trial.suggest_loguniform("dist_fraction", 0.1, 0.8)
    size_threshold = trial.suggest_int("size_threshold", 1, 10)
    beta = trial.suggest_uniform("beta", 0.1, 0.4)
    num_refine_steps = trial.suggest_int("num_refine_steps", 1, 4)
    refine_local_iter = trial.suggest_int("refine_local_iter", 1, 10)
    optimizer_name = trial.suggest_categorical("optimizer_name", ["sgd", "adam"])
    if optimizer_name == "sgd":
        lr = trial.suggest_loguniform("optimizer_params_lr", 1e-3, 1e-1)
        momentum = trial.suggest_uniform("optimizer_params_momentum", 0.0, 0.9)
        optimizer_param_dict = {"lr": lr, "momentum": momentum}
    else:
        lr = trial.suggest_loguniform("optimizer_params_lr", 1e-4, 1e-2)
        optimizer_param_dict = {"lr": lr}
    num_clients_per_round = trial.suggest_int("num_clients_per_round", 1, 10)
    config = {
        "model": {
            "name": "two_layer_lin",
            "params": {
                "input_size": data_config["dataset"]["input_size"],
                "num_classes": data_config["dataset"]["num_classes"],
                "hidden_size": data_config["dataset"]["hidden_size"],
            },
        },
        "dist_fraction": dist_fraction,
        "size_threshold": size_threshold,
        "freq": {"metrics": 30, "save": 150, "print": 100},
        "beta": beta,
        "init": {"iterations": 210},
        "refine": {"local_iter": refine_local_iter},
        "num_refine_steps": num_refine_steps,
        "num_clients_per_round": num_clients_per_round,
        "optimizer": {"name": optimizer_name, "params": optimizer_param_dict},
        "dist_metric": "cross_entropy",
    }
    config = data_config | config

    return config
