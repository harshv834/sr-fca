def get_hp_config(trial):

    model  = {"name" : "stacked_lstm", "params" : {"seq_len": 80, "emb_dim": 8, "n_hidden": 100, "num_classes": 80, "n_layers": 2}}
    stop_threshold = trial.suggest_loguniform("stop_threshold", 0.01, 5)
    client_threshold = trial.suggest_loguniform("client_threshold", 0.01, 5)
    gamma_max = trial.suggest_float("gamma_max", 0, 1)
    num_clusters = trial.suggest_int("num_clusters", 1, 4)
    local_iter = 400
    rounds = 6
    num_clients_per_round = 10
    iterations = 2400
    optimizer = {"name": "sgd", "params": {"lr": 0.8}}
    freq = {"metrics": 30, "save": 150, "print": 60}
    config = {
        "model": model,
        "num_clusters": num_clusters ,
        "local_iter": local_iter,
        "stop_threshold": stop_threshold,
        "client_threshold": client_threshold,
        "gamma_max": gamma_max,
        "num_clients_per_round": num_clients_per_round,
        "num_clusters": num_clusters,
        "freq": freq,
        "iterations" : iterations,
        "rounds" : rounds,
        "optimizer": optimizer,
    }

    return config
