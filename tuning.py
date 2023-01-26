from functools import partialmethod
from time import time

import numpy as np
import yaml
from ray import air, tune
from ray.air import session, RunConfig
from ray.tune import with_parameters, with_resources
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from tqdm import tqdm
import os

from src.clustering import CLUSTERING_DICT
from src.datasets.base import FLDataset
from src.utils import args_getter, check_nan, get_search_space, tune_config_update
import logging
import torch

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
# logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
import warnings
import ray

# warnings.filterwarnings(
#     "ignore", category=pl.utilities.warnings.LightningDeprecationWarning
# )


def objective(config, fldataset):

    clustering = CLUSTERING_DICT[args["clustering"]](
        fldataset.config, tune=True, tune_config=config
    )
    metrics = clustering.cluster(fldataset)
    if fldataset.config["dataset"]["name"] == "synthetic":
        metric_name = "loss"
    else:
        metric_name = "acc"
    if check_nan(metrics):
        # raise ValueError("Nan or inf occurred in metrics")
        session.report({"test_metric": np.nan})
    else:
        session.report({"test_metric": metrics["test"][metric_name]})


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
ray.init(
    address="local",
    log_to_driver=False,
    num_cpus=16,
    num_gpus=torch.cuda.device_count(),
)
tuner = tune.Tuner(
    with_resources(
        with_parameters(objective, fldataset=fldataset),
        resources={"cpu": 5, "gpu": 1},
    ),
    tune_config=tune.TuneConfig(
        search_alg=algo,
        num_samples=args["num_samples"],
    ),
    run_config=RunConfig(
        name=fldataset.config["clustering"] + "_" + fldataset.config["dataset"]["name"],
        local_dir=os.path.join(fldataset.config["path"]["results"], "tuning"),
        verbose=0,
    ),
)
results = tuner.fit()
# ## Check_nan here with config
try:
    best_result = results.get_best_result(metric="test_metric", mode=mode)
    best_result_config = best_result.config
    # best_result_config = tune_config_update(best_result_config)
    print("Best_config", best_result_config)

    with open(best_hp_path, "w") as f:
        yaml.dump(best_result_config, f, default_flow_style=False)
    ## Save metrics for best config as well
except RuntimeError:
    print("All trials ended with test metric NaN")
