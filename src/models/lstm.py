import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class StackedLSTM(BaseModel):
    def __init__(self, config):
        super(StackedLSTM, self).__init__(config)
        self.seq_len = self.config["dataset"]["seq_len"]
        self.num_classes = self.config["dataset"]["num_classes"]
        self.n_hidden = self.config["dataset"]["n_hidden"]
        self.emb_dim = self.config["dataset"]["emb_dim"]
        self.n_layers = self.config["dataset"]["n_layers"]
        self.embedding = nn.Embedding(self.num_classes, self.emb_dim)
        self.rnn = nn.LSTM(self.emb_dim, self.n_hidden, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.n_hidden, self.num_classes)
        self.batch_size = None
        self.hidden = None

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

    #TODO: See this from lightning examples
    def training_step(self, batch, batch_idx):
        (X, Y) = batch
        if self.batch_size is None or self.batch_size != X.shape[0]:
            self.batch_size = X.shape[0]
            self.hidden = self.zero_state(self.batch_size)
        out, self.hidden = self.model(X, self.hidden)
        loss = self.loss_func(out, Y)
        return loss

        
