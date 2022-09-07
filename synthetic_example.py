import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from abc import ABC
from tqdm import tqdm
import torchvision
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


config = {}
config["seed"] = 1231
seed = config["seed"]
os.environ["PYTHONHASHSEED"] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)
config["participation_ratio"] = 1
config["total_num_clients_per_cluster"] = 8
config["num_clients_per_cluster"] = int(
    config["participation_ratio"] * config["total_num_clients_per_cluster"]
)
config["num_clusters"] = 2
config["num_clients"] = config["num_clients_per_cluster"] * config["num_clusters"]
config["dataset"] = "synthetic"
DATASET_LIB = {
    "mnist": torchvision.datasets.MNIST,
    "emnist": torchvision.datasets.EMNIST,
    "cifar10": torchvision.datasets.CIFAR10,
}
config["dataset_dir"] = "./experiments/dataset"
config["results_dir"] = "./experiments/results"
config["results_dir"] = os.path.join(
    config["results_dir"], config["dataset"], "seed_{}".format(seed)
)


class ClientDataset(Dataset):
    def __init__(self, data, transforms=None):
        super(ClientDataset, self).__init__()
        self.data = data[0]
        self.labels = data[1]
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx_data = self.data[idx]
        if self.transforms is not None:
            transformed_data = self.transforms(idx_data)
        else:
            transformed_data = idx_data
        idx_labels = self.labels[idx]
        return (transformed_data.float(), idx_labels)


class Client:
    def __init__(
        self,
        train_data,
        test_data,
        client_id,
        train_transforms,
        test_transforms,
        train_batch_size,
        test_batch_size,
        save_dir,
    ):
        self.trainset = ClientDataset(train_data, train_transforms)
        self.testset = ClientDataset(test_data, test_transforms)
        self.trainloader = DataLoader(
            self.trainset, batch_size=train_batch_size, shuffle=True, num_workers=1
        )
        self.testloader = DataLoader(
            self.testset, batch_size=test_batch_size, shuffle=False, num_workers=1
        )
        self.train_iterator = iter(self.trainloader)
        self.test_iterator = iter(self.testloader)
        self.client_id = client_id
        self.save_dir = os.path.join(save_dir, "init", "client_{}".format(client_id))

    def sample_batch(self, train=True):
        iterator = self.train_iterator if train else self.test_iterator
        try:
            (data, labels) = next(iterator)
        except StopIteration:
            if train:
                self.train_iterator = iter(self.trainloader)
                iterator = self.train_iterator
            else:
                self.test_iterator = iter(self.testloader)
                iterator = self.test_iterator
            (data, labels) = next(iterator)
        return (data, labels)


def generate_data(config):
    d = 30
    w_mid = 10 * np.random.rand(d)
    w_delta = np.random.rand(d)
    w_delta = 10 * w_delta / np.linalg.norm(w_delta)
    w_1 = w_mid + w_delta
    w_2 = w_mid - w_delta
    sigma = 1
    zeta = 10
    num_training_points = 200
    train_data = []
    test_data = []
    w_list = [w_1, w_2]
    for i in range(config["num_clusters"]):
        for _ in range(config["num_clients_per_cluster"]):
            X_train = np.random.rand(num_training_points, d) * zeta
            y_train = (
                X_train @ w_list[i] + np.random.rand(num_training_points, d) * sigma
            )
            train_data.append({"x": X_train, "y": y_train})
            X_test = np.random.rand(num_training_points, d) * zeta
            y_test = X_test @ w_list[i] + np.random.rand(num_training_points, d) * sigma
            test_data.append({"x": X_test, "y": y_test})

    return train_data, test_data


config["train_batch"] = 50
config["test_batch"] = 200
client_loaders = []

train_data, test_data = generate_data(config)
for client_id in range(config["num_clients"]):
    client_loaders.append(
        Client(
            (
                np.array(train_data[client_id]["x"]),
                np.array(train_data[client_id]["y"]),
            ),
            (np.array(test_data[client_id]["x"]), np.array(test_data[client_id]["y"])),
            client_id,
            train_transforms=torchvision.transforms.ToTensor(),
            test_transforms=torchvision.transforms.ToTensor(),
            train_batch_size=config["train_batch"],
            test_batch_size=config["test_batch"],
            save_dir=config["results_dir"],
        )
    )


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(30, 1)

    def forward(self, x):
        return self.fc(x)


def calc_mse(model, device, client_data, train):
    loader = client_data.trainloader if train else client_data.testloader
    model.eval()
    model.to(device)
    sq_err = 0
    tot_num = 0
    with torch.no_grad():
        for (X, Y) in loader:
            X = X.to(device)
            pred = model(X).detach().cpu()
            sq_err += (Y - pred).float().square().sum()
            tot_num += Y.shape[0]
    mse = sq_err / tot_num
    return mse


class BaseTrainer(ABC):
    def __init__(self, config, save_dir):
        super(BaseTrainer, self).__init__()
        self.model = SimpleLinear()
        self.save_dir = save_dir
        self.device = config["device"]
        self.loss_func = LOSSES[config["loss_func"]]
        self.config = config
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def load_model_weights(self):
        model_path = os.path.join(self.save_dir, "model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            print("No model present at path : {}".format())

    def save_model_weights(self):
        model_path = os.path.join(self.save_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)

    def save_metrics(self, train_loss, test_acc, iteration):
        torch.save(
            {"train_loss": train_loss, "test_acc": test_acc},
            os.path.join(self.save_dir, "metrics_{}.pkl".format(iteration)),
        )


class ClientTrainer(BaseTrainer):
    def __init__(self, config, save_dir, client_id):
        super(ClientTrainer, self).__init__(config, save_dir)
        self.client_id = client_id

    def train(self, client_data):
        train_loss_list = []
        test_mse_list = []
        self.model.to(self.device)
        self.model.train()
        optimizer = OPTIMIZER_LIST[self.config["optimizer"]](
            self.model.parameters(), **self.config["optimizer_params"]
        )
        for iteration in tqdm(range(self.config["iterations"])):
            self.model.zero_grad()
            (X, Y) = client_data.sample_batch(train=True)
            X = X.to(self.device)
            Y = Y.to(self.device)
            out = self.model(X)
            loss = self.loss_func(out, Y)
            loss.backward()
            optimizer.step()
            train_loss = loss.detach().cpu().numpy().item()
            train_loss_list.append(train_loss)
            test_mse = calc_mse(self.model, self.device, client_data, train=False)
            test_mse_list.append(test_mse)
            self.model.train()
            if (
                iteration % self.config["save_freq"] == 0
                or iteration == self.config["iterations"] - 1
            ):
                self.save_model_weights()
                self.save_metrics(train_loss_list, test_mse_list, iteration)
            if (
                iteration % self.config["print_freq"] == 0
                or iteration == self.config["iterations"] - 1
            ):
                print(
                    "Iteration : {} \n , Train Loss : {} \n, Test mse : {} \n".format(
                        iteration, train_loss, test_mse
                    )
                )

        self.model.eval()
        self.model.cpu()

    def test(self, client_data):
        self.load_model_weights()
        self.model.eval()
        self.model.to(self.device)
        mse = calc_mse(self.model, client_data)
        self.model.cpu()
        return mse


OPTIMIZER_LIST = {"sgd": optim.SGD, "adam": optim.Adam}
LOSSES = {"cross_entropy": nn.CrossEntropyLoss()}
# config["save_dir"] = os.path.join("./results")
config["iterations"] = 100
config["optimizer_params"] = {"lr": 0.001}
config["save_freq"] = 2
config["print_freq"] = 20
config["model"] = "cnn"
config["optimizer"] = "adam"
config["loss_func"] = "cross_entropy"
# config["model_params"] = {"num_channels": 1 , "num_classes"  : 62}
config["model_params"] = {}
config["device"] = torch.device("cuda:0")
import pickle

client_trainers = [
    ClientTrainer(
        config, os.path.join(config["results_dir"], "init", client_id), client_id
    )
    for client_id in config["selected_clients"]
]


for i in tqdm(range(len(config["selected_clients"]))):
    client_trainers[i].train(client_loaders[i])


import networkx as nx

G = nx.Graph()
G.add_nodes_from(range(config["num_clients"]))
import itertools


def model_weights_diff(w_1, w_2):
    norm_sq = 0
    assert w_1.keys() == w_2.keys(), "Model weights have different keys"
    for key in w_1.keys():
        norm_sq += (w_1[key] - w_2[key]).norm() ** 2
    return np.sqrt(norm_sq)


thresh = 5


all_pairs = list(itertools.combinations(range(config["num_clients"]), 2))
arr = []
for pair in all_pairs:
    w_1 = client_trainers[pair[0]].model.state_dict()
    w_2 = client_trainers[pair[1]].model.state_dict()
    norm_diff = model_weights_diff(w_1, w_2)
    arr.append(norm_diff)


for i in range(len(all_pairs)):
    if arr[i] < thresh:
        G.add_edge(all_pairs[i][0], all_pairs[i][1])
G = G.to_undirected()


clustering = []


def correlation_clustering(G):
    global clustering
    if len(G.nodes) == 0:
        return
    else:
        cluster = []
        new_cluster_pivot = random.sample(G.nodes, 1)[0]
        cluster.append(new_cluster_pivot)
        neighbors = G[new_cluster_pivot].copy()
        for node in neighbors:
            cluster.append(node)
            G.remove_node(node)
        G.remove_node(new_cluster_pivot)
        clustering.append(cluster)
        correlation_clustering(G)


correlation_clustering(G)

clusters = [cluster for cluster in clustering if len(cluster) > 1]
cluster_map = {i: clusters[i] for i in range(len(clusters))}
beta = 0.15


class ClusterTrainer(BaseTrainer):
    def __init__(self, config, save_dir, cluster_id):
        super(ClusterTrainer, self).__init__(config, save_dir)
        self.cluster_id = cluster_id

    def train(self, client_data_list):
        num_clients = len(client_data_list)

        train_loss_list = []
        test_acc_list = []
        self.model.to(self.device)
        self.model.train()

        optimizer = OPTIMIZER_LIST[self.config["optimizer"]](
            self.model.parameters(), **self.config["optimizer_params"]
        )
        # eff_num_workers = int(num_clients/(1 - 2*beta))
        # if eff_num_workers > 0:
        #     eff_batch_size = self.config["train_batch"]/eff_num_workers
        #     for i in range(num_clients):
        #         client_data_list[i].trainloader.batch_size = eff_batch_size

        for iteration in tqdm(range(self.config["iterations"])):
            trmean_buffer = {}
            for idx, param in self.model.named_parameters():
                if param.requires_grad:
                    trmean_buffer[idx] = []
            train_loss = 0
            # optimizer.zero_grad(set_to_none=True)

            for client in client_data_list:
                # if eff_num_workers>0:
                optimizer.zero_grad(set_to_none=True)
                (X, Y) = client.sample_batch()
                X = X.to(config["device"])
                Y = Y.to(config["device"])
                loss_func = nn.CrossEntropyLoss()
                out = self.model(X)
                loss = loss_func(out, Y)
                loss.backward()
                train_loss += loss.detach().cpu().numpy().item()

                with torch.no_grad():
                    for idx, param in self.model.named_parameters():
                        if param.requires_grad:
                            trmean_buffer[idx].append(param.grad.clone())
            train_loss = train_loss / num_clients
            optimizer.zero_grad()

            start_idx = int(beta * num_clients)
            end_idx = int((1 - beta) * num_clients)
            if end_idx <= start_idx + 1:
                start_idx = 0
                end_idx = num_clients

            for idx, param in self.model.named_parameters():
                if param.requires_grad:
                    sorted, _ = torch.sort(
                        torch.stack(trmean_buffer[idx], dim=0), dim=0
                    )
                    new_grad = sorted[start_idx:end_idx, ...].mean(dim=0)
                    param.grad = new_grad
                    trmean_buffer[idx] = []
            optimizer.step()

            train_loss_list.append(train_loss)
            test_acc = 0
            for client_data in client_data_list:
                test_acc += calc_mse(self.model, self.device, client_data, train=False)
            test_acc = test_acc / num_clients
            test_acc_list.append(test_acc)
            self.model.train()
            if (
                iteration % self.config["save_freq"] == 0
                or iteration == self.config["iterations"] - 1
            ):
                self.save_model_weights()
                self.save_metrics(train_loss_list, test_acc_list, iteration)
            if (
                iteration % self.config["print_freq"] == 0
                or iteration == self.config["iterations"] - 1
            ):
                print(
                    "Iteration : {} \n , Train Loss : {} \n, Test Acc : {} \n".format(
                        iteration, train_loss, test_acc
                    )
                )

        self.model.eval()
        self.model.cpu()

    def test(self, client_data_list):
        self.load_model_weights()
        self.model.eval()
        self.model.to(self.device)
        test_acc = 0
        for client_data in client_data_list:
            test_acc += calc_mse(self.model, self.device, client_data, train=False)
        self.model.cpu()
        return test_acc


def avg_mse(model_wts, client_data_list):
    orig = model_wts[0]
    if len(model_wts) > 0:
        for wt in model_wts[1:]:
            for key in orig.keys():
                if orig[key].dtype == torch.float32:
                    orig[key] += wt[key]
        for key in orig.keys():
            if orig[key].dtype == torch.float32:
                orig[key] = orig[key] / len(model_wts)
    model = SimpleLinear()
    model.load_state_dict(orig)
    model.to(memory_format=torch.channels_last).cuda()
    test_acc = 0
    for client_data in client_data_list:
        test_acc += calc_mse(model, torch.device("cuda:0"), client_data, train=False)
    test_acc = test_acc / len(client_data_list)
    return test_acc, orig


config["refine_steps"] = 2

for refine_step in tqdm(range(config["refine_steps"])):
    beta = 0.15
    cluster_trainers = []
    for cluster_id in tqdm(cluster_map.keys()):
        cluster_clients = [client_loaders[i] for i in cluster_map[cluster_id]]
        cluster_trainer = ClusterTrainer(
            config,
            os.path.join(
                config["results_dir"],
                "refine_{}".format(refine_step),
                "cluster_{}".format(cluster_id),
            ),
            cluster_id,
        )
        cluster_trainer.train(cluster_clients)
        cluster_trainers.append(cluster_trainer)
    with open(
        os.path.join(
            config["results_dir"], "refine_{}".format(refine_step), "cluster_maps.pkl"
        ),
        "wb",
    ) as handle:
        pickle.dump(cluster_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    cluster_map_recluster = {}
    for key in cluster_map.keys():
        cluster_map_recluster[key] = []

    for i in tqdm(range(config["num_clients"])):
        w_node = client_trainers[i].model.state_dict()
        norm_diff = np.infty
        new_cluster_id = 0
        for cluster_id in cluster_map.keys():
            w_cluster = cluster_trainers[cluster_id].model.state_dict()
            curr_norm_diff = model_weights_diff(w_node, w_cluster)
            if norm_diff > curr_norm_diff:
                new_cluster_id = cluster_id
                norm_diff = curr_norm_diff

        cluster_map_recluster[new_cluster_id].append(i)
    keys = list(cluster_map_recluster.keys()).copy()
    for key in keys:
        if len(cluster_map_recluster[key]) == 0:
            cluster_map_recluster.pop(key)
    cluster_map = cluster_map_recluster

    G = nx.Graph()
    G.add_nodes_from(cluster_map.keys())

    all_pairs = list(itertools.combinations(cluster_map.keys(), 2))
    for pair in tqdm(all_pairs):
        w_1 = cluster_trainers[pair[0]].model.state_dict()
        w_2 = cluster_trainers[pair[1]].model.state_dict()
        norm_diff = model_weights_diff(w_1, w_2)
        if norm_diff < thresh:
            G.add_edge(pair[0], pair[1])
    G = G.to_undirected()
    clustering = []
    correlation_clustering(G)
    merge_clusters = [cluster for cluster in clustering if len(cluster) > 0]

    # merge_cluster_map = {i: clusters[i] for i in range(len(clusters))}
    # clusters = list(nx.algorithms.clique.enumerate_all_cliques(G))
    cluster_map_new = {}
    for i in range(len(merge_clusters)):
        cluster_map_new[i] = []
        for j in merge_clusters[i]:
            cluster_map_new[i] += cluster_map[j]
    cluster_map = cluster_map_new
    test_acc = 0
    for cluster_id in tqdm(cluster_map.keys()):
        cluster_clients = [client_loaders[i] for i in cluster_map[cluster_id]]
        model_wts = [
            cluster_trainers[j].model.state_dict() for j in merge_clusters[cluster_id]
        ]
        test_acc_cluster, model_avg_wt = avg_mse(model_wts, cluster_clients)
        torch.save(
            model_avg_wt,
            os.path.join(
                config["results_dir"],
                "refine_{}".format(refine_step),
                "merged_cluster_{}.pth".format(cluster_id),
            ),
        )
        test_acc += test_acc_cluster
    test_acc = test_acc / len(cluster_map.keys())
    torch.save(
        test_acc,
        os.path.join(
            config["results_dir"], "refine_{}".format(refine_step), "avg_acc.pth"
        ),
    )


class GlobalTrainer(BaseTrainer):
    def __init__(self, config, save_dir):
        super(GlobalTrainer, self).__init__(config, save_dir)

    def train(self, client_data_list):
        num_clients = len(client_data_list)

        train_loss_list = []
        test_acc_list = []
        self.model.to(self.device)
        self.model.train()

        optimizer = OPTIMIZER_LIST[self.config["optimizer"]](
            self.model.parameters(), **self.config["optimizer_params"]
        )
        # eff_num_workers = int(num_clients/(1 - 2*beta))
        # if eff_num_workers > 0:
        #     eff_batch_size = self.config["train_batch"]/eff_num_workers
        #     for i in range(num_clients):
        #         client_data_list[i].trainloader.batch_size = eff_batch_size

        for iteration in tqdm(range(self.config["iterations"])):
            trmean_buffer = {}
            for idx, param in self.model.named_parameters():
                if param.requires_grad:
                    trmean_buffer[idx] = []
            train_loss = 0
            # optimizer.zero_grad(set_to_none=True)

            for client in client_data_list:
                # if eff_num_workers>0:
                optimizer.zero_grad(set_to_none=True)
                (X, Y) = client.sample_batch()
                X = X.to(config["device"])
                Y = Y.to(config["device"])
                loss_func = nn.CrossEntropyLoss()
                out = self.model(X)
                loss = loss_func(out, Y)
                loss.backward()
                train_loss += loss.detach().cpu().numpy().item()

                with torch.no_grad():
                    for idx, param in self.model.named_parameters():
                        if param.requires_grad:
                            trmean_buffer[idx].append(param.grad.clone())
            train_loss = train_loss / num_clients
            optimizer.zero_grad()

            start_idx = 0
            end_idx = num_clients

            for idx, param in self.model.named_parameters():
                if param.requires_grad:
                    sorted, _ = torch.sort(
                        torch.stack(trmean_buffer[idx], dim=0), dim=0
                    )
                    new_grad = sorted[start_idx:end_idx, ...].mean(dim=0)
                    param.grad = new_grad
                    trmean_buffer[idx] = []
            optimizer.step()

            train_loss_list.append(train_loss)
            test_acc = 0
            for client_data in client_data_list:
                test_acc += calc_mse(self.model, self.device, client_data, train=False)
            test_acc = test_acc / num_clients
            test_acc_list.append(test_acc)
            self.model.train()
            if (
                iteration % self.config["save_freq"] == 0
                or iteration == self.config["iterations"] - 1
            ):
                self.save_model_weights()
                self.save_metrics(train_loss_list, test_acc_list, iteration)
            if (
                iteration % self.config["print_freq"] == 0
                or iteration == self.config["iterations"] - 1
            ):
                print(
                    "Iteration : {} \n , Train Loss : {} \n, Test Acc : {} \n".format(
                        iteration, train_loss, test_acc
                    )
                )

        self.model.eval()
        self.model.cpu()

    def test(self, client_data_list):
        self.load_model_weights()
        self.model.eval()
        self.model.to(self.device)
        test_acc = 0
        for client_data in client_data_list:
            test_acc += calc_mse(self.model, self.device, client_data, train=False)
        self.model.cpu()
        return test_acc


global_trainer = GlobalTrainer(config, os.path.join(config["results_dir"], "global"))
global_trainer.train(client_loaders)

# class IFCATrainer(BaseTrainer):
    
    
# class MOCHATrainer(BaseTrainer):
    
# class SattlerTrainer(BaseTrainer):
    