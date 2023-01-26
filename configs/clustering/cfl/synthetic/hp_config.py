def get_hp_config(trial):
    model = {"name": "one_layer_lin", "params":{"dimension": 1000,"scale": 1}}
    stop_threshold = trial.suggest_loguniform("stop_threshold", 0.001, 100)
    client_threshold = trial.suggest_loguniform("client_threshold", 0.001, 100)
    gamma_max = trial.suggest_float("gamma_max", 0, 1)
    num_clusters = 2
    local_iter = 5
    rounds = 56
    iterations = 280
    num_clients_per_round = 10
    optimizer = {"name": "sgd","params":{"lr" : 0.001}}
    freq = {"metrics": 30,"print": 30,"save": 60}
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
        "iterations": iterations,
        "rounds" : rounds,
        "optimizer": optimizer,
    }

    return config