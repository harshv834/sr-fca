def get_hp_config(trial):
    dist_threshold = trial.suggest_float("dist_threshold", 0.23931624328323814, 1.4720229728078364)
    size_threshold = 2
    beta = trial.suggest_float("beta", 0.05, 0.35)
    model  = {"name" : "stacked_lstm", "params" : {"seq_len": 80, "emb_dim": 8, "n_hidden": 100, "num_classes": 80, "n_layers": 2}}
    local_iter = 400
    rounds = 6
    iterations = 2400
    num_refine_steps=trial.suggest_int("num_refine_steps", 1, 2)
    dist_metric = "cross_entropy"
    num_clients_per_round = 10
    optimizer = {"name": "sgd", "params": {"lr": 0.8}}
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
