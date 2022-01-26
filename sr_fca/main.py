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





config = {}
config["num_clients"] = 20
config["dataset"] = "mnist"
DATASET_LIB = {"emnist" : torchvision.datasets.MNIST, "emnist": torchvision.datasets.EMNIST, "cifar10": torchvision.datasets.CIFAR10}
config["dataset_dir"] = "./experiments/dataset"
config["results_dir"] = "./experiments/results"
config["results_dir"] = os.path.join(config["results_dir"], config["datasets"])

os.makedirs(config["results_dir"], exist_ok=True)

def split(dataset_size, num_clients):
    split_idx = []
    all_idx = np.arange(dataset_size)
    for client in range(num_clients):
        split_idx.append(all_idx[all_idx%client == 0])
    return split_idx

def dataset_split(train_data, test_data, num_clients):
    train_size = train_data[0].shape[0]
    train_split_idx = split(train_size, num_clients)
    train_chunks = [(train_data[0][train_split_idx[client]], train_data[1][train_split_idx[client]]) for client in range(num_clients)]
    test_size = test_data[0].shape[0]
    test_split_idx = split(test_size, num_clients)
    test_chunks = [(test_data[0][test_split_idx[client]], test_data[1][test_split_idx[client]]) for client in range(num_clients)]
    return train_chunks, test_chunks

def make_client_datasets(config, num_clusters = 10):
    train_dataset = DATASET_LIB[config["dataset"]](root = config['dataset_dir'], download = True, train=True, split="byclass")
    test_dataset = DATASET_LIB[config["dataset"]](root = config['dataset_dir'], download = True, train=False, split = "byclass")

    train_data = (train_dataset.train_data, train_dataset.train_labels)
    test_data = (test_dataset.test_data, test_dataset.test_labels)
    train_chunks, test_chunks = dataset_split(train_data, test_data, config["num_clients"])
    num_clusters = 2
    return train_chunks, test_chunks


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
        self.trainloader = DataLoader(self.trainset, batch_size = train_batch_size, shuffle=True)
        self.testloader = DataLoader(self.testset, batch_size = test_batch_size, shuffle=False)
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
client_loaders = [Client(train_chunks[i], test_chunks[i], i, train_transforms=None, test_transforms=None, train_batch_size=50, test_batch_size=512, save_dir=config["save_dir"]) for i in range(config["num_clients"])]

class SimpleCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels,32, kernel_size=(5,5), padding="same")
        self.pool1 = nn.MaxPool2d((2,2), stride = 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5,5), padding="same")
        self.pool2 = nn.MaxPool2d((2,2), stride = 2)
        self.fc1 = nn.Linear(7*7*64, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

    
    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
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


  
MODEL_LIST = {"cnn" : SimpleCNN}
OPTIMIZER_LIST = {"sgd": optim.SGD, "adam": optim.Adam}
LOSSES = {"cross_entropy": nn.CrossEntropyLoss()}
config["save_dir"] = os.path.join("./results")
config["iterations"] = 4
config["optimizer_params"] = {"lr":0.001}
config["save_freq"] = 2
config["model"] = "cnn"
config["optimizer"] = "adam"
config["loss_func"] = "cross_entropy"
config["model_params"] = {"num_channels": 1 , "num_classes"  : 62}
config["device"] = torch.device("cuda:0")
import pickle
client_trainers = [ClientTrainer(config,os.path.join(config["save_dir"], "init", "node_{}".format(i)), i) for i in range(config["num_clients"])]


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
thresh = 39
all_pairs = list(itertools.combinations(range(config["num_clients"]),2))
for pair in all_pairs:
    w_1  = client_trainers[pair[0]].model.state_dict()
    w_2 = client_trainers[pair[1]].model.state_dict()
    norm_diff = model_weights_diff(w_1, w_2)
    if norm_diff < thresh:
        G.add_edge(pair[0], pair[1])
        
cliques = list(nx.algorithms.clique.enumerate_all_cliques(G))

t = 4
clusters = [clique  for clique in cliques if len(clique) > 5]
cluster_map = {i: clusters[i] for i in range(len(clusters))}
cluster_id = 0
beta = 0.2
cluster_clients = [client_loaders[i] for i in cluster_map[cluster_id]]

class ClusterTrainer(BaseTrainer):
    def __init__(self,  config, save_dir,cluster_id):
        super(ClusterTrainer, self).__init__(config, save_dir)
        self.cluster_id = cluster_id
    
    def train(self, client_data_list):
        num_clients = len(client_data_list)

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []
        self.model.to(self.device)
        self.model.train()
        optimizer = OPTIMIZER_LIST[self.config["optimizer"]](self.model.parameters(), **self.config["optimizer_params"])
        for iteration in tqdm(range(self.config["iterations"])):

            trmean_buffer = {}
            for idx, param in self.model.named_parameters():
                trmean_buffer[idx] = []
            train_loss = 0
            for client in client_data_list:
                self.optimizer.zero_grad()
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

            self.optimizer.zero_grad()
            for idx, param in self.model.named_parameters():
                sorted, _  = torch.sort(torch.stack(trmean_buffer[idx], dim=0), dim=0)
                new_grad = sorted[int(beta*num_clients)):int((1-beta)*num_clients)),...].mean(dim=0)
                param.grad = new_grad
                trmean_buffer[idx] = []
            optimizer.step()
            
            train_loss_list.append(train_loss/num_clients)
            test_acc = 0
            for client_data in client_data_list:
                test_acc += calc_acc(self.model, self.device, client_data, train=False)
            test_acc_list.append(test_acc/num_clients)
            self.model.train()
            if iteration % self.config["save_freq"] == 0 or iteration == self.config["iterations"] - 1:
                self.save_model_weights()
                self.save_metrics(train_loss_list, test_acc_list)
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
refine_step = 0
cluster_trainer = ClusterTrainer(config, os.path.join(config['save_dir'], "refine_{}".format(refine_step), "cluster_{}".format(cluster_id)), cluster_id)


# def main():
    
    
    
# if __name__ == "__main__":
#     main()