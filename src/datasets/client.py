import os
from torch.utils.data import Dataset, DataLoader


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
        return (transformed_data, idx_labels)


class Client:
    def __init__(self, batch_size, client_dataset, client_id):
        train_data, test_data = client_dataset
        self.trainset = ClientDataset(train_data)
        self.testset = ClientDataset(test_data)
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size["train"],
            shuffle=True,
            num_workers=1,
        )
        self.testloader = DataLoader(
            self.testset,
            batch_size=batch_size["test"],
            shuffle=False,
            num_workers=1,
        )
        self.train_iterator = iter(self.trainloader)
        self.test_iterator = iter(self.testloader)
        self.client_id = client_id

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


# def make_data_chunks(train_data, ):

# def generate_data(config):
#     d = 30
#     w_delta = np.random.rand(d)
#     w_delta = w_delta / np.linalg.norm(w_delta)
#     w_1 = 2 * w_delta
#     w_2 = -2 * w_delta
#     sigma = 0.05
#     zeta = 1
#     num_training_points = 100
#     train_data = []
#     test_data = []
#     w_list = [w_1, w_2]
#     for i in range(config["num_clusters"]):
#         for _ in range(config["num_clients_per_cluster"]):
#             X_train = np.random.rand(num_training_points, d) * zeta
#             y_train = X_train @ w_list[i] + np.random.rand(num_training_points) * sigma
#             train_data.append({"x": torch.Tensor(X_train), "y": torch.Tensor(y_train)})
#             X_test = np.random.rand(num_training_points, d) * zeta
#             y_test = X_test @ w_list[i] + np.random.rand(num_training_points) * sigma
#             test_data.append({"x": torch.Tensor(X_test), "y": torch.Tensor(y_test)})

#     return train_data, test_data
