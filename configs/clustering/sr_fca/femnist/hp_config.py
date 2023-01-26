def get_hp_config(trial):
    dist_threshold = trial.suggest_float("dist_threshold", 0.026177632431516232, 0.5548499828010662)
    size_threshold = 2
    beta = trial.suggest_float("beta", 0, 0.4)
    model  = {"name" : "simplecnn"}
    local_iter = 5
    rounds = 200
    iterations = 1000
    num_refine_steps= trial.suggest_int("num_refine_steps", 1, 2)
    dist_metric = "cross_entropy"
    num_clients_per_round = 10
    optimizer = {"name": "sgd", "params": {"lr": 0.04}}
    freq = {"metrics": 30, "save": 150, "print": 60}
    config = {
        "model": model,
        "local_iter": local_iter,
        "num_clients_per_round": num_clients_per_round,
        "freq": freq,
        "rounds" : rounds,
        "optimizer": optimizer,
        "init":{
            "iterations" : iterations
        },
        "refine" : {
            "local_iter" : local_iter,
            "rounds" : rounds
        },
        "beta" : beta,
        "size_threshold" : size_threshold,
        "dist_threshold": dist_threshold,
        "num_refine_steps": num_refine_steps,
        "dist_metric" : dist_metric,
    }

    return config

    