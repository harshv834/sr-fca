import logging
from functools import partialmethod
from time import time

import numpy as np

# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
import pytorch_lightning as pl
import torch
import yaml
from ray import air, tune
from ray.air import session
from ray.tune import with_parameters, with_resources
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from tqdm import tqdm

from src.clustering import CLUSTERING_DICT
from src.datasets.base import FLDataset
from src.utils import args_getter, check_nan, get_search_space, tune_config_update

logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
import warnings

warnings.filterwarnings("ignore")


def objective(config, fldataset):

    clustering = CLUSTERING_DICT[args["clustering"]](
        fldataset.config, tune=True, tune_config=config
    )
    clustering.cluster(fldataset)
    metrics = clustering.final_metrics
    if fldataset.config["dataset"]["name"] == "synthetic":
        metric_name = "loss"
    else:
        metric_name = "acc"
    if check_nan(metrics):
        # raise ValueError("Nan or inf occurred in metrics")
        session.report({"test_metric": np.nan})
    else:
        session.report({"test_metric": metrics["test"][metric_name].item()})


t0 = time()
print("here")
args = args_getter()

args["time"] = {"t0": t0}
fldataset = FLDataset(args, tune=True)
print("FL Dataset created in {} s".format(fldataset.config["time"]["tdataset"] - t0))

if args["dataset"] == "synthetic":
    mode = "min"
else:
    mode = "max"
best_hp_path, search_space_func = get_search_space(fldataset.config)
searcher = OptunaSearch(space=search_space_func, metric="test_metric", mode=mode)
algo = ConcurrencyLimiter(searcher, max_concurrent=4)
tuner = tune.Tuner(
    with_resources(
        with_parameters(objective, fldataset=fldataset),
        tune.PlacementGroupFactory([{"cpu": 8, "gpu": 2}]),
    ),
    tune_config=tune.TuneConfig(
        search_alg=algo,
        num_samples=1,
    ),
)
results = tuner.fit()
# ## Check_nan here with config
try:
    best_result = results.get_best_result(metric="test_metric", mode=mode)
    best_result_config = results.get_best_result().config

    best_result_config = tune_config_update(best_result_config)

    print("Best_config", best_result.config)

    with open(best_hp_path, "w") as f:
        yaml.dump(results.get_best_result().config, f, default_flow_style=False)
    ## Save metrics for best config as well
except RuntimeError:
    print("All trials ended with test metric NaN")
