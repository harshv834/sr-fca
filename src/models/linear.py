import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OneLayer(nn.Module):
    def __init__(self, dimension, scale):
        super(OneLayer, self).__init__()
        self.fc = nn.Linear(dimension, 1, bias=False)
        self.fc.weight.data = (
            torch.tensor(
                np.random.binomial(1, 0.5, size=(1, dimension)).astype(np.float32)
            )
            * scale
        )

    def forward(self, x):
        return self.fc(x).view(-1)


class TwoLayer(nn.Module):
    def __init__(self, input_size=784, hidden_size=2048, num_classes=10):
        super(TwoLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
