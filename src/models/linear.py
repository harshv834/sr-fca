import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class OneLayer(BaseModel):
    def __init__(self, config):
        super(OneLayer, self).__init__(config)
        self.dimension = self.config["dataset"]["dimension"]
        self.scale = self.config["dataset"]["scale"]
        self.fc = nn.Linear(self.dimension, 1, bias=False)
        self.fc.weight.data = (
            torch.tensor(
                np.random.binomial(1, 0.5, size=(1, self.dimension)).astype(np.float32)
            )
            * self.scale
        )

    def forward(self, x):
        return self.fc(x).view(-1)


class TwoLayer(BaseModel):
    def __init__(self, config):
        super(TwoLayer, self).__init__(config)
        self.input_size = self.config["dataset"]["input_size"]
        self.hidden_size = self.config["dataset"]["hidden_size"]
        self.num_classes = self.config["dataset"]["num_classes"]
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
