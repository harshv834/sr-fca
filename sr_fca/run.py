import argparse
import logging

import flwr as fl
from .utils import load_config, set_seed,

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--cluster",
    type=str,
    default="sr-fca",
    choices=["local", "global", "sr-fca", "ifca"],
    help="Clustering algo",
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    default="./dataset/rot_mnist.json",
    help="Config for FL dataset",
)
parser.add_argument(
    "-t", "--trial", type=int, default=0, help="id for recording runs and setting seeds"
)
parser.add_argument(
    "-s", "--seed", type=int, default=42, help="random seed for experiments"
)


args = parser.parse_args()

# Set logging
logging.basicConfig( 
    format="[%(levelname)s][%(asctime)s]: %(message)s",
    level=getattr(logging, args.log.upper()),
    datefmt="%H:%M:%S",
)


def main():

    ## Read configs
    config = load_config(args)

    ## Set seed
    set_seed(args.seed)

    ## Initialize other stuff --- tensorboard, model_parameters, gpu, paths

    ## Load data
    
    clients = load_data(config)

    ## Initialize clients
    
    strategy = select_strategy(config)

    ## Perform algorithm

    fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=config.NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    )