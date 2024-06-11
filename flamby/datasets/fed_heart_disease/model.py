import torch
import torch.nn as nn

class Baseline_old(nn.Module):
    def __init__(self, dropout, input_dim=18, output_dim=1):
        super(Baseline_old, self).__init__()
        self.dropout = dropout
        if self.dropout == 0:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.layer1 = nn.Linear(input_dim, int(input_dim / 2))
            self.dropout_layer = nn.Dropout(p=self.dropout)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(int(input_dim / 2), output_dim)

    def forward(self, x):
        x = x.type(torch.float32)
        if self.dropout == 0:
            return torch.sigmoid(self.linear(x))
        else:
            x = self.layer1(x)
            x = self.relu(x)
            x = self.dropout_layer(x)
            x = self.layer2(x)
            x = torch.sigmoid(x)
            return x


class Baseline(nn.Module):
    def __init__(self, dropout, input_dim=18, output_dim=2):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(13, 32)
        # self.fc2 = nn.Linear(32, 64)
        # self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = x.type(torch.float32)
        x = self.act(self.fc1(x))
        # x = self.act(self.fc2(x))
        # x = self.act(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
