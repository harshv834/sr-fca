def get_hp_config(trial):
    optimizer_name = trial.suggest_categorical("optimizer_name", ["sgd", "adam"])
    if optimizer_name == "sgd":
        lr = trial.suggest_loguniform("optimizer_params_lr", 1e-4, 1e-2)
        optimizer_param_dict = {"lr": lr}
    else:
        lr = trial.suggest_loguniform("optimizer_params_lr", 5e-5, 1e-3)
        optimizer_param_dict = {"lr": lr}
    optimizer = {"name" : optimizer_name, "params" : optimizer_param_dict}
    num_clients_per_round = trial.suggest_int("num_clients_per_round", 1, 10)

    local_iter = 5
    rounds = 50
    num_clients_per_round = 10
    params = {"input_size": 784, "hidden_size": 200, "num_classes": 10}
    freq = {"metrics": 30, "save": 150, "print": 60}
    iterations = 250
    config = {
        "model": {
            "name": "two_layer_lin",
            "params": params,
        },
        "local_iter": local_iter,
        "num_clients_per_round": num_clients_per_round,
        "freq": freq,
        "iterations": iterations,
        "rounds" : rounds,
        "optimizer": optimizer,
    }

    return config