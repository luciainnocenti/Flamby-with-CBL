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
        self.fc = nn.Linear(input_dim, output_dim)

        self.layer1 = nn.Linear(input_dim, int(input_dim/2))
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(int(input_dim/2), output_dim)

    def forward(self, x):
        if self.dropout == 0:
            return self.fc(x)
        else:
            x = self.relu(self.layer1(x))
            x = self.dropout_layer(x)
            x = self.layer2(x)
            return x


if __name__ == "__main__":

    mydataset = FedTcgaBrca(train=True, pooled=True)

    model = Baseline()

    for i in range(10):
        X = torch.unsqueeze(mydataset[i][0], 0)
        y = torch.unsqueeze(mydataset[i][1], 0)
        print(X.shape)
        print(y)
        print(model(X))



'''
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
            x = self.dropout_layer(x)
            x = self.layer2(x)
            return torch.sigmoid(x)

'''
