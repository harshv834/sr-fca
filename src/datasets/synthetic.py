import os

import numpy as np
import torch


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
        dataset_config = config["dataset"]
        num_clusters = dataset_config["num_clusters"]
        num_clients = config["num_clients"]
        num_samples = dataset_config["num_samples"]

        dimension = dataset_config["dimension"]
        scale = dataset_config["scale"]
        assert (
            num_clients % num_clusters == 0
        ), "{} clients cannot be distributed into {} clusters".format(
            num_clients, num_clusters
        )

        w_cluster_list = [
            np.random.binomial(1, 0.5, size=(dimension)).astype("float32") * scale
            for _ in range(num_clusters)
        ]

        train_chunks, test_chunks = [], []
        for i in range(num_clients):
            w_star = w_cluster_list[i % num_clusters]
            x_train = np.random.randn(num_samples["train"], dimension).astype("float32")
            y_train = (
                x_train @ w_star
                + np.random.randn(num_samples["train"]) * dataset_config["noise_scale"]
            )
            train_chunks.append((x_train, y_train))

            x_test = np.random.randn(num_samples["test"], dimension).astype("float32")
            y_test = (
                x_test @ w_star + np.random.randn(num_samples["test"])
            ) * dataset_config["noise_scale"]

            test_chunks.append((x_test, y_test))
        torch.save(
            {"train": train_chunks, "test": test_chunks},
            os.path.join(dataset_path, "data.pth"),
        )

    return train_chunks, test_chunks
