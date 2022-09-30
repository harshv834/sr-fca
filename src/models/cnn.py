import torch.nn as nn

import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), padding="same")
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), padding="same")
        self.pool2 = nn.MaxPool2d((2, 2), stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 62)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
