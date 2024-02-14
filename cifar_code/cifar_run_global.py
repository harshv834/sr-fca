import numpy as np
import os
from cifar_trainers import GlobalTrainer
from cifar_utils import create_argparse, create_config
from cifar_dataset import create_client_loaders

parser = create_argparse()


def main(args):
    config = create_config(args)
    client_loaders = create_client_loaders(config)

    # rounds = 2400 // 10
    # local_iter = 10
    # config["iterations"] = 10
    

    global_trainer = GlobalTrainer(config, os.path.join(config["results_dir"], "global"))

    client_loaders = np.array(client_loaders)
    iters_per_epoch = 50000 // int(
        config["train_batch"] * config["total_num_clients_per_cluster"]
    )
    epochs = 2400 // iters_per_epoch
    lr_schedule = np.interp(
        np.arange((epochs + 1) * iters_per_epoch),
        [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
        [0, 1, 0],
    )
    global_trainer.train(client_loaders, lr_schedule)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
