import torch
import torch.nn as nn

from flamby.datasets.fed_tcga_brca import FedTcgaBrca


class Baseline(nn.Module):
    """
    Baseline model: a linear layer !
    """

    def __init__(self, dropout):
        super(Baseline, self).__init__()
        input_dim = 39
        output_dim = 1
        self.dropout = dropout
        if self.dropout == 0:
            self.fc = nn.Linear(input_dim, output_dim)
        else:
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)
            self.act = nn.LeakyReLU()
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.dropout == 0:
            return self.fc(x)
        else:
            x = x.type(torch.float32)
            x = self.act(self.fc1(x))
            x = self.dropout(x)
            x = self.act(self.fc2(x))
            x = self.dropout(x)
            x = self.act(self.fc3(x))
            x = self.fc4(x)
            return x
