def get_hp_config(trial):
    local_iter = 5
    rounds = 56
    iterations = 280
    model = {"name" : "one_layer_lin", "params" : {"dimension": 1000, "scale":  1}}
    optimizer_name = trial.suggest_categorical("optimizer_name", ["sgd", "adam"])
    if optimizer_name == "sgd":
        lr = trial.suggest_loguniform("optimizer_params_lr", 1e-4, 1e-2)
        # momentum = trial.suggest_uniform("optimizer_params_momentum", 0.0, 0.9)
        optimizer_param_dict = {"lr": lr}
    else:
        lr = trial.suggest_loguniform("optimizer_params_lr", 1e-4, 1e-2)
        optimizer_param_dict = {"lr": lr}
    num_clients_per_round = 10
    num_clusters = 2
    optimizer = {"name": optimizer_name, "params" : optimizer_param_dict}
    freq = {"metrics": 30, "save": 30, "print": 60}
    config = {
        "model": model,
        "num_clusters": num_clusters,
        "local_iter": local_iter,
        "rounds" : rounds,
        "iterations" : iterations,
        "num_clients_per_round": num_clients_per_round,
        "freq": freq,
        "optimizer": optimizer,
    }

    return config
