import torch.nn as nn


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(30, 1)

    def forward(self, x):
        return self.fc(x).view(-1)
