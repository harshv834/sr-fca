from functools import partialmethod
from time import time

from tqdm import tqdm

from src.clustering import CLUSTERING_DICT
from src.datasets.base import FLDataset
from src.utils import args_getter
import torch

#tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
import logging

import pytorch_lightning as pl

#logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
import warnings

#warnings.filterwarnings("ignore")
if __name__ == "__main__":
    t0 = time()
    args = args_getter()
    args["time"] = {"t0": t0}
    fldataset = FLDataset(args)
    print(
        "FL Dataset created in {} s".format(fldataset.config["time"]["tdataset"] - t0)
    )
    trainer = pl.Trainer(
        default_root_dir=fldataset.config["path"]["results"],
        devices=torch.cuda.device_count(),
        accelerator="gpu",
        enable_model_summary=False,
        enable_progress_bar=False,
        strategy="ddp_spawn",
        precision=16,
        amp_backend="native",
        log_every_n_steps=1,
    )

    clustering = CLUSTERING_DICT[args["clustering"]](fldataset.config, tune=False)
    metrics = clustering.cluster(experiment=fldataset, trainer=trainer)
    print(
        "Clustered FL ran in {} s".format(
            clustering.config["time"]["tcluster"]
            - clustering.config["time"]["tdataset"]
        )
    )
    print(metrics)
