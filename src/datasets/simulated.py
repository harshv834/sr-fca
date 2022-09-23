import numpy as np
from torchvision.datasets import MNIST, CIFAR10

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


def dataset_split(train_data, test_data, num_clients, shuffle):
    train_size = train_data[0].shape[0]
    train_split_idx = split(train_size, num_clients, shuffle)
    train_chunks = [
        (
            train_data[0][train_split_idx[client].tolist()],
            np.array(train_data[1])[train_split_idx[client].tolist()].tolist(),
        )
        for client in range(num_clients)
    ]
    test_size = test_data[0].shape[0]
    test_split_idx = split(test_size, num_clients, shuffle)
    test_chunks = [
        (
            test_data[0][test_split_idx[client]].tolist(),
            np.array(test_data[1])[test_split_idx[client].tolist()].tolist(),
        )
        for client in range(num_clients)
    ]
    return train_chunks, test_chunks


def apply_client_het(train_chunks, test_chunks, transformation, num_clusters):
    assert (
        len(train_chunks) % num_clusters == 0
    ), "Number of clients {} not divisible by number of clusters {}".format(
        len(train_chunks), num_clusters
    )
    if transformation == "inv":
        assert num_clusters == 2, "Inversion can create 2 clusters only"
        raise NotImplementedError

    elif transformation == "rot":
        raise NotImplementedError
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
        train_chunks, test_chunks, client_het, config["num_clusters"]
    )
    return train_chunks, test_chunks
