def get_hp_config(trial, data_config):
    stop_threshold = trial.suggest_loguniform("stop_threshold", 0.1, 100)
    client_threshold = trial.suggest_loguniform("client_threshold", 0.1, 100)
    gamma_max = trial.suggest_float("gamma_max", 0, 1)

    local_iter = trial.suggest_int("local_iter", 1, 5)
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
            "name": "two_layer_lin",
            "params": {
                "input_size": data_config["dataset"]["input_size"],
                "num_classes": data_config["dataset"]["num_classes"],
                "hidden_size": data_config["dataset"]["hidden_size"],
            },
        },
        "num_clusters": num_clusters,
        "local_iter": local_iter,
        "stop_threshold": stop_threshold,
        "client_threshold": client_threshold,
        "gamma_max": gamma_max,
        "num_clients_per_round": num_clients_per_round,
        "num_clusters": num_clusters,
        "freq": {"metrics": 30, "save": 150, "print": 100},
        "iterations": 210,
        "optimizer": {"name": optimizer_name, "params": optimizer_param_dict},
    }
    config = data_config | config

    return config
