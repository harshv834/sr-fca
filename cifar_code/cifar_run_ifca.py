import torch
import numpy as np
from tqdm import tqdm
import os


from cifar_trainers import GlobalTrainer
from cifar_utils import create_argparse, create_config, calc_loss
from cifar_dataset import create_client_loaders

parser = create_argparse()




def determine_clustering(ifca_trainers, client_data_list):

    cluster_map = {0: [], 1: []}
    for i in range(16):
        loss_list = []
        for j in range(2):
            loss_list.append(
                calc_loss(
                    ifca_trainers[j].model, torch.device("cuda:0"), client_data_list[i], train=True
                )
            )
        if loss_list[0] > loss_list[1]:
            cluster_map[1].append(i)
        else:
            cluster_map[0].append(i)
    return cluster_map


def main(args):
    config = create_config(args)
    client_loaders = create_client_loaders(config)

    rounds = 2400
    config["iterations"] = 1
    
    ## Initial cluster_map
    all_clients = list(range(16))
    np.random.shuffle(all_clients)
    cluster_map = {0: all_clients[:8], 1: all_clients[8:]}


    client_loaders = np.array(client_loaders)
    ifca_trainers = [
        GlobalTrainer(config, os.path.join(config["results_dir"], "ifca", "cluster_0")),
        GlobalTrainer(
            config,
            os.path.join(
                config["results_dir"],
                "ifca",
                "cluster_1",
            ),
        ),
    ]
    iters_per_epoch = 50000 // int(
        config["train_batch"] * config["total_num_clients_per_cluster"]
    )
    epochs = 2400 // iters_per_epoch
    lr_schedule = np.interp(
        np.arange((epochs + 1) * iters_per_epoch),
        [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
        [0, 1, 0],
    )
    metrics = {0:{"train_loss" : [], "test_acc":  []}, 1:{"train_loss" : [], "test_acc" : []}}
    for round_id in tqdm(range(rounds)):
        for i in range(2):
            if len(cluster_map[i]) > 0:
                train_loss_list, test_acc_list = ifca_trainers[i].train(
                    client_loaders[cluster_map[i]],
                    lr_schedule[
                        round_id: round_id + 2
                    ],
                )
                metrics[i]["train_loss"] += train_loss_list
                metrics[i]["test_acc"] += test_acc_list
            # else:
            #     print("Cluster {} has 0 clients in round {}".format(i, round_id))
        cluster_map = determine_clustering(ifca_trainers, client_loaders)
        if round_id % config["print_freq"] == 0 or round_id == rounds - 1:
            print("Curr cluster_map : {}".format(cluster_map))
            print("Curr test metrics : cluster 0 : {}, cluster  1 : {}".format(metrics[0]["test_acc"][-1], metrics[1]["test_acc"][-1]))
            
        if round_id % config["save_freq"] == 0 or round_id == rounds - 1:

            torch.save(
                cluster_map,
                os.path.join(
                    config["results_dir"], "ifca", "cluster_map.pth"
                ),
            )
            torch.save(metrics, os.path.join(config["results_dir"], "ifca", "metrics.pth"))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
