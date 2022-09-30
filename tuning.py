from lib2to3.pgen2.token import OP
import torch
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune import with_parameters, with_resources
from src.utils import args_getter
from src.datasets.base import FLDataset
from src.clustering import CLUSTERING_DICT
from tqdm import tqdm
from functools import partialmethod
from time import time
import numpy as np
from src.utils import check_nan, get_search_space
import yaml

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


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
# from ray.util import inspect_serializability

# print(inspect_serializability(with_parameters(objective, fldataset=fldataset)))
# import ipdb

# ipdb.set_trace()
tuner = tune.Tuner(
    with_resources(
        with_parameters(objective, fldataset=fldataset), resources={"cpu": 8, "gpu": 1}
    ),
    tune_config=tune.TuneConfig(
        search_alg=algo,
        num_samples=20,
    ),
    run_config=air.RunConfig(failure_config=air.FailureConfig(fail_fast=True)),
)
results = tuner.fit()
# ## Check_nan here with config
results_df = results.get_dataframe()(metric="test_metric", mode=mode)
print("Best_config", results.get_best_result().config)
with open(best_hp_path, "w") as f:
    yaml.dump(results.get_best_result().config, f, default_flow_style=False)
## Save metrics for best config as well
