import os
import torch
import numpy as np


def generate_synthetic_data(config, dataset_path):
    """Generate synthetic data if not present load it again

    Args:
        config (dict): _description_
        dataset_path (str): _description_

    Returns:
        _type_: _description_
    """
    data_path = os.path.join(dataset_path, "data.pth")
    if os.path.exists(data_path):
        data = torch.load(data_path)
        train_chunks, test_chunks = data["train"], data["test"]
    else:
        num_clusters = config["num_clusters"]
        num_clients = config["num_clients"]
        num_samples = config["num_samples"]

        dimension = config["dimension"]
        scale = config["scale"]
        assert (
            num_clients % num_clusters == 0
        ), "{} clients cannot be distributed into {} clusters".format(
            num_clients, num_clusters
        )

        w_cluster_list = [
            torch.tensor(
                np.random.binomial(1, 0.5, size=(dimension)).astype(np.float32)
            )
            * scale
            for _ in range(num_clusters)
        ]

        train_chunks, test_chunks = [], []
        for i in range(num_clients):
            w_star = w_cluster_list[i % num_clusters]
            x_train = torch.randn((num_samples["train"], dimension))
            y_train = (
                x_train @ w_star
                + torch.randn((num_samples["train"])) * config["noise_scale"]
            )
            train_chunks.append((x_train, y_train))

            x_test = torch.randn((num_samples["test"], dimension))
            y_test = (
                x_test @ w_star
                + torch.randn((num_samples["test"])) * config["noise_scale"]
            )
            test_chunks.append((x_test, y_test))
        torch.save(
            {"train": train_chunks, "test": test_chunks},
            os.path.join(dataset_path, "data.pth"),
        )

    return train_chunks, test_chunks
