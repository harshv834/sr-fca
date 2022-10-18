import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.datasets import CIFAR10, MNIST

DATASET_LIB = {
    "mnist": MNIST,
    "cifar10": CIFAR10,
}


def split(dataset_size, num_clients):
    split_idx = []
    all_idx = np.arange(dataset_size)
    all_idx = np.random.permutation(all_idx)
    split_idx = np.array_split(all_idx, num_clients)
    return split_idx


def dataset_split(train_data, test_data, num_clients):
    train_size = train_data[0].shape[0]
    train_split_idx = split(train_size, num_clients)
    train_chunks = [
        (
            train_data[0][train_split_idx[client].tolist()],
            np.array(train_data[1])[train_split_idx[client].tolist()].tolist(),
        )
        for client in range(num_clients)
    ]
    test_size = test_data[0].shape[0]
    test_split_idx = split(test_size, num_clients)
    test_chunks = [
        (
            test_data[0][test_split_idx[client].tolist()],
            np.array(test_data[1])[test_split_idx[client].tolist()].tolist(),
        )
        for client in range(num_clients)
    ]
    return train_chunks, test_chunks


def apply_client_het(train_chunks, test_chunks, transformation, num_clusters):
    num_clients = len(train_chunks)
    # assert (
    #     num_clients % num_clusters == 0
    # ), "Number of clients {} not divisible by number of clusters {}".format(
    #     len(train_chunks), num_clusters
    # )
    if transformation == "inv":
        assert num_clusters == 2, "Inversion can create 2 clusters only"
        for i in range(num_clients):
            if i % num_clusters == 1:
                train_chunks[i] = (
                    (
                        255 * torch.ones(train_chunks[i][0].shape) - train_chunks[i][0]
                    ).byte(),
                    train_chunks[i][1],
                )
                test_chunks[i] = (
                    (
                        255 * torch.ones(test_chunks[i][0].shape) - test_chunks[i][0]
                    ).byte(),
                    test_chunks[i][1],
                )

    elif transformation == "rot":
        return train_chunks, test_chunks       
        # assert num_clusters in [2, 4], "Currently support only 2 or 4 rotated clusters"
        # for i in range(num_clients):
        #     theta = int(360 * (i % num_clusters) / num_clusters)
        #     train_chunks[i] = (rot_img(train_chunks[i][0], theta), train_chunks[i][1])

    else:
        raise ValueError(
            "Invalid transformation {}, existing choices are inv and rot".format(
                transformation
            )
        )
    return train_chunks, test_chunks


def load_simulated_dataset(dataset_name, client_het, dataset_path, config):
    data_source = DATASET_LIB[dataset_name]
    train_dataset = data_source(root=dataset_path, download=True, train=True)
    test_dataset = data_source(root=dataset_path, download=True, train=False)
    train_data = (train_dataset.data, train_dataset.targets)
    test_data = (test_dataset.data, test_dataset.targets)
    train_chunks, test_chunks = dataset_split(
        train_data, test_data, config["num_clients"]
    )
    train_chunks, test_chunks = apply_client_het(
        train_chunks, test_chunks, client_het, config["dataset"]["num_clusters"]
    )
    return train_chunks, test_chunks


def rot_img(x, theta):
    if len(x.shape) == 4:
        x = x.transpose(0, 3, 1, 2)
    numpy_arr = False
    if type(x) == np.ndarray:
        x = torch.tensor(x)
        numpy_arr = True
    x = TF.rotate(x, theta)
    if numpy_arr:
        x = x.numpy()
    if len(x.shape) == 4:
        x = x.transpose(0, 2, 3, 1)

    return x
