



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
config["seed"] = 42
seed = config["seed"]
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

config["participation_ratio"] = 0.1
config["total_num_clients_per_cluster"] = 80
config["num_clients_per_cluster"] = int(config["participation_ratio"]*config["total_num_clients_per_cluster"])
config["num_clusters"] = 4
config["num_clients"] = config["num_clients_per_cluster"]*config["num_clusters"]
config["dataset"] = "mnist"
DATASET_LIB = {"mnist" : torchvision.datasets.MNIST, "emnist": torchvision.datasets.EMNIST, "cifar10": torchvision.datasets.CIFAR10}
config["dataset_dir"] = "./experiments/dataset"
config["results_dir"] = "./experiments/results"
config["results_dir"] = os.path.join(config["results_dir"], config["dataset"], "seed_{}".format(seed))

os.makedirs(config["results_dir"], exist_ok=True)

def split(dataset_size, num_clients, shuffle):
    split_idx = []
    all_idx = np.arange(dataset_size)
    if shuffle:
        all_idx = np.random.permutation(all_idx)
    split_idx = np.array_split(all_idx, num_clients)
    return split_idx

def dataset_split(train_data, test_data, num_clients, shuffle):
    train_size = train_data[0].shape[0]
    train_split_idx = split(train_size, num_clients, shuffle)
    train_chunks = [(train_data[0][train_split_idx[client]], train_data[1][train_split_idx[client]]) for client in range(num_clients)]
    test_size = test_data[0].shape[0]
    test_split_idx = split(test_size, num_clients, shuffle)
    test_chunks = [(test_data[0][test_split_idx[client]], test_data[1][test_split_idx[client]]) for client in range(num_clients)]
    return train_chunks, test_chunks

def make_client_datasets(config):
    train_chunks_total = []
    test_chunks_total = []
    train_dataset = DATASET_LIB[config["dataset"]](root = config['dataset_dir'], download = True, train=True)
    test_dataset = DATASET_LIB[config["dataset"]](root = config['dataset_dir'], download = True, train=False)

    train_data = (train_dataset.data, train_dataset.targets)
    test_data = (test_dataset.data, test_dataset.targets)
    for i in range(config["num_clusters"]):
        train_chunks, test_chunks = dataset_split(train_data, test_data, config["total_num_clients_per_cluster"], shuffle=True)
        chunks_idx = np.random.choice(np.arange(len(train_chunks)), size=config["num_clients_per_cluster"], replace=False).astype(int)
        train_chunks = np.array(train_chunks)[chunks_idx].tolist()
        test_chunks = np.array(test_chunks)[chunks_idx].tolist()
        train_chunks_total += train_chunks
        test_chunks_total += test_chunks
    return train_chunks_total, test_chunks_total


class ClientDataset(Dataset):
    def __init__(self, data,transforms = None):
        super(ClientDataset,self).__init__()
        self.data = data[0]
        self.labels = data[1]
        self.transforms = transforms

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        idx_data = self.data[idx]
        if self.transforms is not None:
            transformed_data =self.transforms(idx_data)
        else:
            transformed_data = idx_data
        idx_labels = self.labels[idx]
        return (transformed_data.unsqueeze(0).float(), idx_labels)

class Client():
    def __init__(self, train_data, test_data, client_id,  train_transforms, test_transforms, train_batch_size, test_batch_size, save_dir):
        self.trainset = ClientDataset(train_data, train_transforms)
        self.testset = ClientDataset(test_data, test_transforms)
        self.trainloader = DataLoader(self.trainset, batch_size = train_batch_size, shuffle=True, num_workers=4)
        self.testloader = DataLoader(self.testset, batch_size = test_batch_size, shuffle=False, num_workers=4)
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
train_chunks, test_chunks = make_client_datasets(config)

config["train_batch"] = 128
config["test_batch"] = 512
client_loaders = []
for i in range(config["num_clusters"]):
    for j in range(config["num_clients_per_cluster"]):
        idx = i * config["num_clients_per_cluster"] + j
        x_train = train_chunks[idx][0]
        x_test = test_chunks[idx][0]
        if i >0:
            x_train = torch.rot90(x_train, i, [1, 2])
            x_test = torch.rot90(x_test, i, [1, 2])
        client_loaders.append(
            Client(
                (x_train, train_chunks[idx][1]),
                (x_test, test_chunks[idx][1]),
                idx,
                train_transforms=None,
                test_transforms=None,
                train_batch_size=config["train_batch"],
                test_batch_size=config["test_batch"],
                save_dir=config["results_dir"],
            )
        )
class SimpleLinear(torch.nn.Module):

    def __init__(self, h1=2048):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, h1)
        self.fc2 = torch.nn.Linear(h1, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    
def calc_acc(model, device, client_data, train):
    loader = client_data.trainloader if train else client_data.testloader
    model.eval()
    acc = 0
    with torch.no_grad():
        for (X,Y) in loader:
            X = X.to(device)
            pred = model(X).argmax(axis=1).detach().cpu()
            acc += (Y == pred).float().mean()
    acc = acc/len(loader)
    acc *= 100.0
    return acc


class BaseTrainer(ABC):
    def __init__(self,config, save_dir):
        super(BaseTrainer, self).__init__()
        self.model = MODEL_LIST[config["model"]](**config["model_params"])
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
        model_path  = os.path.join(self.save_dir, "model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            print("No model present at path : {}".format())

    def save_model_weights(self):
        model_path  = os.path.join(self.save_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)
    def save_metrics(self, train_loss, test_acc):
        torch.save({"train_loss": train_loss,  "test_acc" : test_acc}, os.path.join(self.save_dir,"metrics.pkl"))

class ClientTrainer(BaseTrainer):
    def __init__(self,  config, save_dir,client_id):
        super(ClientTrainer, self).__init__(config, save_dir)
        self.client_id = client_id
    
    def train(self, client_data):
        train_loss_list = []
        test_acc_list = []
        self.model.to(self.device)
        self.model.train()
        optimizer = OPTIMIZER_LIST[self.config["optimizer"]](self.model.parameters(), **self.config["optimizer_params"])
        for iteration in tqdm(range(self.config["iterations"])):
            self.model.zero_grad()
            (X,Y) = client_data.sample_batch(train=True)
            X = X.to(self.device)
            Y = Y.to(self.device)
            out = self.model(X)
            loss = self.loss_func(out, Y)
            loss.backward()
            optimizer.step()
            train_loss = loss.detach().cpu().numpy().item()
            train_loss_list.append(train_loss)
            test_acc = calc_acc(self.model, self.device, client_data, train=False)
            test_acc_list.append(test_acc)
            self.model.train()
            if iteration % self.config["save_freq"] == 0 or iteration == self.config["iterations"] - 1:
                self.save_model_weights()
                self.save_metrics(train_loss_list, test_acc_list)
            if iteration % self.config["print_freq"] == 0 or iteration == self.config["iterations"] - 1:
                print("Iteration : {} \n , Train Loss : {} \n, Test Acc : {} \n".format(iteration,  train_loss, test_acc))
                
        self.model.eval()
        self.model.cpu()


    def test(self, client_data):
        self.load_model_weights()
        self.model.eval()
        self.model.to(self.device)
        acc =  calc_acc(self.model, client_data)
        self.model.cpu()
        return acc


  
MODEL_LIST = {"lin" : SimpleLinear}
OPTIMIZER_LIST = {"sgd": optim.SGD, "adam": optim.Adam}
LOSSES = {"cross_entropy": nn.CrossEntropyLoss()}
# config["save_dir"] = os.path.join("./results")
config["iterations"] = 80
config["optimizer_params"] = {"lr":0.001}
config["save_freq"] = 2
config["print_freq"]  = 20
config["model"] = "lin"
config["optimizer"] = "adam"
config["loss_func"] = "cross_entropy"
#config["model_params"] = {"num_channels": 1 , "num_classes"  : 62}
config["model_params"] = {}
config["device"] = torch.device("cuda:0")
import pickle
client_trainers = [ClientTrainer(config,os.path.join(config["results_dir"], "init", "node_{}".format(i)), i) for i in range(config["num_clients"])]


for i in tqdm(range(config["num_clients"])):
    client_trainers[i].train(client_loaders[i])
    




import networkx as nx
G = nx.Graph()
G.add_nodes_from(range(config["num_clients"]))
import itertools
def model_weights_diff(w_1, w_2):
    norm_sq = 0
    assert w_1.keys() == w_2.keys(), "Model weights have different keys"
    for key in w_1.keys():
        norm_sq  += (w_1[key] - w_2[key]).norm()**2
    return np.sqrt(norm_sq)
wt = client_trainers[0].model.state_dict()
# thresh = 0
# for key in wt.keys():
#     thresh += wt[key].norm()**2
# print(torch.sqrt(thresh))
# thresh = 37.68

all_pairs = list(itertools.combinations(range(config["num_clients"]),2))
arr = []
for pair in all_pairs:
    w_1  = client_trainers[pair[0]].model.state_dict()
    w_2 = client_trainers[pair[1]].model.state_dict()
    norm_diff = model_weights_diff(w_1, w_2)
    arr.append(norm_diff)
#thresh = torch.mean(torch.tensor(arr))
thresh = arr[torch.tensor(arr).argsort()[int(0.3*len(arr))-1]]
for i in range(len(all_pairs)):
    if arr[i] < thresh:
        G.add_edge(all_pairs[i][0], all_pairs[i][1])
G = G.to_undirected()
#cliques = list(nx.algorithms.clique.enumerate_all_cliques(G))


clustering = []
def correlation_clustering(G):
    global clustering
    if len(G.nodes) == 0:
        return
    else:
        cluster = []
        new_cluster_pivot = random.sample(G.nodes,1)[0]
        cluster.append(new_cluster_pivot)
        neighbors = G[new_cluster_pivot].copy()
        for node in neighbors:
            cluster.append(node)
            G.remove_node(node)
        G.remove_node(new_cluster_pivot)
        clustering.append(cluster)
        correlation_clustering(G)
correlation_clustering(G)


#config["t"] = 7
#t = config["t"]
clusters = [cluster  for cluster in clustering if len(clustering) > 1 ]
cluster_map = {i: clusters[i] for i in range(len(clusters))}
beta = 0.2


class ClusterTrainer(BaseTrainer):
    def __init__(self,  config, save_dir,cluster_id):
        super(ClusterTrainer, self).__init__(config, save_dir)
        self.cluster_id = cluster_id
    
    def train(self, client_data_list):
        num_clients = len(client_data_list)

        train_loss_list = []
        test_acc_list = []
        self.model.to(self.device)
        self.model.train()
        
        
        optimizer = OPTIMIZER_LIST[self.config["optimizer"]](self.model.parameters(), **self.config["optimizer_params"])
        #eff_num_workers = int(num_clients/(1 - 2*beta))
        # if eff_num_workers > 0:
        #     eff_batch_size = self.config["train_batch"]/eff_num_workers
        #     for i in range(num_clients):
        #         client_data_list[i].trainloader.batch_size = eff_batch_size
                
        for iteration in tqdm(range(self.config["iterations"])):
            trmean_buffer = {}
            for idx, param in self.model.named_parameters():
                trmean_buffer[idx] = []
            train_loss = 0
            #optimizer.zero_grad(set_to_none=True)

            for client in client_data_list:
                #if eff_num_workers>0:
                optimizer.zero_grad(set_to_none=True)
                (X,Y) = client.sample_batch()
                X = X.to(config["device"])
                Y = Y.to(config["device"])
                loss_func = nn.CrossEntropyLoss()
                out = self.model(X)
                loss = loss_func(out,Y)
                loss.backward()
                train_loss += loss.detach().cpu().numpy().item()
                
                with torch.no_grad():
                    for idx, param in self.model.named_parameters():
                        trmean_buffer[idx].append(param.grad.clone())
            train_loss = train_loss/num_clients
            optimizer.zero_grad()
            
            start_idx = int(beta*num_clients)
            end_idx = int((1-beta)*num_clients)
            if end_idx <= start_idx + 1:
                start_idx = 0
                end_idx = num_clients


            for idx, param in self.model.named_parameters():
                sorted, _  = torch.sort(torch.stack(trmean_buffer[idx], dim=0), dim=0)
                new_grad = sorted[start_idx:end_idx,...].mean(dim=0)
                param.grad = new_grad
                trmean_buffer[idx] = []
            optimizer.step()
            
            train_loss_list.append(train_loss)
            test_acc = 0
            for client_data in client_data_list:
                test_acc += calc_acc(self.model, self.device, client_data, train=False)
            test_acc = test_acc/num_clients
            test_acc_list.append(test_acc)
            self.model.train()
            if iteration % self.config["save_freq"] == 0 or iteration == self.config["iterations"] - 1:
                self.save_model_weights()
                self.save_metrics(train_loss_list, test_acc_list)
            if iteration % self.config["print_freq"] == 0 or iteration == self.config["iterations"] - 1:
                print("Iteration : {} \n , Train Loss : {} \n, Test Acc : {} \n".format(iteration,  train_loss, test_acc))
                
        self.model.eval()
        self.model.cpu()


    def test(self, client_data_list):
        self.load_model_weights()
        self.model.eval()
        self.model.to(self.device)
        test_acc = 0
        for client_data in client_data_list:
            test_acc += calc_acc(self.model, self.device, client_data, train=False)
        self.model.cpu()
        return test_acc


config["refine_steps"] = 2
for refine_step in tqdm(range(config["refine_steps"])):
    beta = 0.3
    cluster_trainers = []
    for cluster_id in tqdm(cluster_map.keys()):
        cluster_clients = [client_loaders[i] for i in cluster_map[cluster_id]]
        cluster_trainer = ClusterTrainer(config, os.path.join(config['results_dir'], "refine_{}".format(refine_step), "cluster_{}".format(cluster_id)), cluster_id)
        cluster_trainer.train(cluster_clients)
        cluster_trainers.append(cluster_trainer)
    with open(os.path.join(config["results_dir"],"refine_{}".format(refine_step), "cluster_maps.pkl"), 'wb') as handle:
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

    all_pairs = list(itertools.combinations(cluster_map.keys(),2))
    for pair in tqdm(all_pairs):
        w_1  = cluster_trainers[pair[0]].model.state_dict()
        w_2 = cluster_trainers[pair[1]].model.state_dict()
        norm_diff = model_weights_diff(w_1, w_2)
        if norm_diff < thresh:
            G.add_edge(pair[0], pair[1])
    G = G.to_undirected()
    clustering = []        
    correlation_clustering(G)
    merge_clusters = [cluster  for cluster in clustering if len(cluster) > 0]
    
    #merge_cluster_map = {i: clusters[i] for i in range(len(clusters))}
    #clusters = list(nx.algorithms.clique.enumerate_all_cliques(G))
    cluster_map_new = {}
    for i in range(len(merge_clusters)):
        cluster_map_new[i] = []
        for j in merge_clusters[i]:
            cluster_map_new[i] += cluster_map[j]
    cluster_map = cluster_map_new


class GlobalTrainer(BaseTrainer):
    def __init__(self,  config, save_dir):
        super(GlobalTrainer, self).__init__(config, save_dir)
        
    def train(self, client_data_list):
        num_clients = len(client_data_list)

        train_loss_list = []
        test_acc_list = []
        self.model.to(self.device)
        self.model.train()
        
        
        optimizer = OPTIMIZER_LIST[self.config["optimizer"]](self.model.parameters(), **self.config["optimizer_params"])
        #eff_num_workers = int(num_clients/(1 - 2*beta))
        # if eff_num_workers > 0:
        #     eff_batch_size = self.config["train_batch"]/eff_num_workers
        #     for i in range(num_clients):
        #         client_data_list[i].trainloader.batch_size = eff_batch_size
                
        for iteration in tqdm(range(self.config["iterations"])):
            trmean_buffer = {}
            for idx, param in self.model.named_parameters():
                trmean_buffer[idx] = []
            train_loss = 0
            #optimizer.zero_grad(set_to_none=True)

            for client in client_data_list:
                #if eff_num_workers>0:
                optimizer.zero_grad(set_to_none=True)
                (X,Y) = client.sample_batch()
                X = X.to(config["device"])
                Y = Y.to(config["device"])
                loss_func = nn.CrossEntropyLoss()
                out = self.model(X)
                loss = loss_func(out,Y)
                loss.backward()
                train_loss += loss.detach().cpu().numpy().item()
                
                with torch.no_grad():
                    for idx, param in self.model.named_parameters():
                        trmean_buffer[idx].append(param.grad.clone())
            train_loss = train_loss/num_clients
            optimizer.zero_grad()
            
            start_idx = 0
            end_idx = num_clients


            for idx, param in self.model.named_parameters():
                sorted, _  = torch.sort(torch.stack(trmean_buffer[idx], dim=0), dim=0)
                new_grad = sorted[start_idx:end_idx,...].mean(dim=0)
                param.grad = new_grad
                trmean_buffer[idx] = []
            optimizer.step()
            
            train_loss_list.append(train_loss)
            test_acc = 0
            for client_data in client_data_list:
                test_acc += calc_acc(self.model, self.device, client_data, train=False)
            test_acc = test_acc/num_clients
            test_acc_list.append(test_acc)
            self.model.train()
            if iteration % self.config["save_freq"] == 0 or iteration == self.config["iterations"] - 1:
                self.save_model_weights()
                self.save_metrics(train_loss_list, test_acc_list)
            if iteration % self.config["print_freq"] == 0 or iteration == self.config["iterations"] - 1:
                print("Iteration : {} \n , Train Loss : {} \n, Test Acc : {} \n".format(iteration,  train_loss, test_acc))
                
        self.model.eval()
        self.model.cpu()


    def test(self, client_data_list):
        self.load_model_weights()
        self.model.eval()
        self.model.to(self.device)
        test_acc = 0
        for client_data in client_data_list:
            test_acc += calc_acc(self.model, self.device, client_data, train=False)
        self.model.cpu()
        return test_acc

global_trainer = GlobalTrainer(config, os.path.join(config["results_dir"], "global"))
global_trainer.train(client_loaders)