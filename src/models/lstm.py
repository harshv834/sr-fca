import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedLSTM(nn.Module):
    def __init__(self, seq_len=80, num_classes=80, n_hidden=256, emb_dim=8, n_layers=2):
        super(StackedLSTM, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.emb_dim = emb_dim
        self.n_layers = 2
        self.embedding = nn.Embedding(self.num_classes, self.emb_dim)
        self.rnn = nn.LSTM(self.emb_dim, self.n_hidden, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.n_hidden, num_classes)

    def forward(self, x, hidden):  # x (n_samples, seq_len)
        x = self.embedding(x)  # x(n_samples, seq_len, emb_dim)
        x, hidden = self.rnn(x, hidden)  # (n_samples, seq_len, n_hidden)
        x = self.fc(x[:, -1, :])
        return x, hidden

    def zero_state(self, batch_size):
        return (
            torch.zeros(self.n_layers, batch_size, self.n_hidden),
            torch.zeros(self.n_layers, batch_size, self.n_hidden),
        )
