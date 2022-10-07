import logging
from functools import partialmethod
from time import time

import numpy as np

from ray_lightning.tune import get_tune_resources
import pytorch_lightning as pl
import torch
import yaml
import ray
from ray import tune
from ray.air.config import RunConfig
from ray.air import session
from ray.tune import with_parameters, with_resources, TuneConfig, Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from tqdm import tqdm
from functools import partialmethod

from src.clustering import CLUSTERING_DICT
from src.datasets.base import FLDataset
from src.utils import args_getter, check_nan, get_search_space, tune_config_update

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
import warnings

warnings.filterwarnings("ignore")


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
searcher = OptunaSearch(
    space=search_space_func,
    metric="test_metric",
    mode=mode,
    seed=fldataset.config["seed"],
)
algo = ConcurrencyLimiter(searcher, max_concurrent=4)
ray.init(
    address="local",
    configure_logging=True,
    logging_level=logging.CRITICAL,
)
analysis = tune.run(
    with_parameters(objective, fldataset=fldataset),
    search_alg=algo,
    num_samples=1,
    name=fldataset.config["dataset"]["name"]
    + "_"
    + fldataset.config["clustering"]
    + "_"
    + str(fldataset.config["seed"]),
    metric="test_metric",
    mode=mode,
    resources_per_trial=get_tune_resources(
        num_cpus_per_worker=3, num_workers=3, use_gpu=True
    ),
    verbose=False,
)
# ## Check_nan here with config
try:
    best_result = analysis.best_result
    print("Best Result", best_result)
    best_result_config = analysis.get_best_config(metric="test_metric", mode=mode)

    best_result_config = tune_config_update(best_result_config)

    print("Best_config", best_result_config)

    with open(best_hp_path, "w") as f:
        yaml.dump(best_result_config, f, default_flow_style=False)
    ## Save metrics for best config as well
except RuntimeError:
    print("All trials ended with test metric NaN")
