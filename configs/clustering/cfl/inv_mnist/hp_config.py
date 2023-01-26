def get_hp_config(trial):
    stop_threshold = trial.suggest_loguniform("stop_threshold", 0.01, 5)
    client_threshold = trial.suggest_loguniform("client_threshold", 0.01, 5)
    gamma_max = trial.suggest_float("gamma_max", 0, 1)
    num_clusters = 2
    local_iter = 5
    rounds = 56
    num_clients_per_round = 10
    optimizer = {"name": "adam", "params": {"lr": 0.0002}}
    params = {"input_size": 784, "hidden_size": 200, "num_classes": 10}
    freq = {"metrics": 30, "save": 150, "print": 60}
    iterations = 280
    config = {
        "model": {
            "name": "two_layer_lin",
            "params": params,
        },
        "num_clusters": num_clusters ,
        "local_iter": local_iter,
        "stop_threshold": stop_threshold,
        "client_threshold": client_threshold,
        "gamma_max": gamma_max,
        "num_clients_per_round": num_clients_per_round,
        "num_clusters": num_clusters,
        "freq": {"metrics": 30, "save": 150, "print": 100},
        "iterations": 280,
        "rounds" : 56,
        "optimizer": optimizer,
    }

    return config
