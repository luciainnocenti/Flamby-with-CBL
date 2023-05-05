import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, input_dim, dropout, output_dim=1):
        super(Baseline, self).__init__()
        self.dropout = dropout
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.layer1 = nn.Linear(input_dim, int(input_dim/2))
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(int(input_dim/2), output_dim)

    def forward(self, x):
        if self.dropout == 0:
            return torch.sigmoid(self.linear(x))
        else:
            x = self.relu(self.layer1(x))
            x = self.dropout(x)
            x = self.layer2(x)
            return torch.sigmoid(x)


'''
class Baseline(nn.Module):
    def __init__(self, dropout, input_dim=13, output_dim=1):
        super(Baseline, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer1 = nn.Linear(input_dim, int(input_dim/2))
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(int(input_dim/2), output_dim)

    def forward(self, x):
        if self.dropout == 0:
            return torch.sigmoid(self.linear(x))
        else:
            x = self.relu(self.layer1(x))
            x = self.dropout(x)
            x = self.layer2(x)
            return x
            '''