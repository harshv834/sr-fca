import json
import os
import random

import numpy as np

from src.language_utils import letter_to_vec, word_to_indices
from src.utils import read_data


def get_shakespeare(config):
    full_data_path = config["full_data_path"]
    data_path = config["path"]["data"]
    train_path = os.path.join(full_data_path, "train")
    test_path = os.path.join(full_data_path, "test")
    all_client_idx, _, train_data, test_data = read_data(train_path, test_path)
    selected_clients_path = os.path.join(data_path, "selected_clients.json")
    if os.path.exists(selected_clients_path):
        with open(selected_clients_path, "r") as f:
            client_idx = json.load(f)

    else:
        if len(all_client_idx) >= config["num_clients"]:
            client_idx = random.sample(all_client_idx, config["num_clients"])
        else:
            client_idx = all_client_idx

        with open(selected_clients_path, "w") as f:
            json.dump(client_idx, f)
    train_chunks = [
        (np.array([word_to_indices(word) for word in train_data[client_id]["x"]]), np.array([letter_to_vec(l) for l in train_data[client_id]["y"]]))
        for client_id in client_idx
    ]
    test_chunks = [
        (np.array([word_to_indices(word) for word in test_data[client_id]["x"]]), np.array([letter_to_vec(l) for l in test_data[client_id]["y"]]))
        for client_id in client_idx
    ]
    return train_chunks, test_chunks, client_idx

    