def get_hp_config(trial):
    stop_threshold = trial.suggest_loguniform("stop_threshold", 0.01, 5)
    client_threshold = trial.suggest_loguniform("client_threshold", 0.01, 5)
    gamma_max = trial.suggest_float("gamma_max", 0, 1)
    num_clusters = trial.suggest_int("num_clusters", 2, 5)
    local_iter = 5
    rounds = 200
    num_clients_per_round = 10
    optimizer = {"name": "sgd", "params": {"lr": 0.04}}
    freq = {"metrics": 30, "save": 150, "print": 60}
    iterations = 1000
    config = {
        "model": {
            "name": "simplecnn",
        },
        "num_clusters": num_clusters ,
        "local_iter": local_iter,
        "stop_threshold": stop_threshold,
        "client_threshold": client_threshold,
        "gamma_max": gamma_max,
        "num_clients_per_round": num_clients_per_round,
        "num_clusters": num_clusters,
        "freq": freq,
        "iterations": iterations,
        "rounds" : rounds,
        "optimizer": optimizer,
    }

    return config
