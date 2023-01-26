def get_hp_config(trial, data_config):
    local_iter = trial.suggest_int("local_iter", 1, 10)
    optimizer_name = trial.suggest_categorical("optimizer_name", ["sgd", "adam"])
    if optimizer_name == "sgd":
        lr = trial.suggest_loguniform("optimizer_params_lr", 1e-3, 1e-1)
        momentum = trial.suggest_uniform("optimizer_params_momentum", 0.0, 0.9)
        optimizer_param_dict = {"lr": lr, "momentum": momentum}
    else:
        lr = trial.suggest_loguniform("optimizer_params_lr", 1e-4, 1e-2)
        optimizer_param_dict = {"lr": lr}
    num_clients_per_round = trial.suggest_int("num_clients_per_round", 1, 10)
    num_clusters = trial.suggest_int("num_clusters", 2, 8)

    config = {
        "model": {
            "name": "one_layer_lin",
            "params": {
                "dimension": data_config["dataset"]["dimension"],
                "scale": data_config["dataset"]["scale"],
            },
        },
        "num_clusters": num_clusters,
        "local_iter": local_iter,
        "num_clients_per_round": num_clients_per_round,
        "freq": {"metrics": 30, "save": 150, "print": 100},
        "iterations": 280,
        "optimizer": {"name": optimizer_name, "params": optimizer_param_dict},
    }
    config = data_config | config

    return config
