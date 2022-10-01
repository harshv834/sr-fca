from src.utils import args_getter
from src.datasets.base import FLDataset
from src.clustering import CLUSTERING_DICT
from tqdm import tqdm
from functools import partialmethod
from time import time

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# ## Find the correct import statement for this
# class Runner(ABC):
#     def __init__(self,args):
#         super(self, Runner).__init__()
#         self.args = args
#         self.logger = None
#         ## Change this since every run has only logs and the experiment name
#         logging.info("Running with args :")
#         logging.info(args)
#         #This can be improved

#     def run(self):

#     ## Parse the arguments for bookkeeping (Create results folders,init logger, etc)
#     self.args = prepare_run(self.args)

#     ## Now that run is prepared, split dataset and create clients
#     client_dataset = create_dataset()
#     self.clients = create_clients(self.args, client_dataset)
#     base_fl, dist_metric, model =
#     clustering_algo = create_clustering_algo()
#     self.results  = clustering_algo(self.clients)
#     self.logger.info(self.results)

if __name__ == "__main__":
    t0 = time()
    args = args_getter()
    args["time"] = {"t0": t0}
    fldataset = FLDataset(args)
    print(
        "FL Dataset created in {} s".format(fldataset.config["time"]["tdataset"] - t0)
    )
    clustering = CLUSTERING_DICT[args["clustering"]](fldataset.config, tune=False)
    metrics = clustering.cluster(fldataset)
    print(
        "Clustered FL ran in {} s".format(
            clustering.config["time"]["tcluster"]
            - clustering.config["time"]["tdataset"]
        )
    )
    print(metrics)

    # cluster_algo_runner = ClusterFLAlgo(args.algorithm)
    # cluster_algo_runner.run(setting)

    # runner = Runner(args)
    # output = runner.run()
    # logging.info("Run complete")
    # logging.info("Arguments)
