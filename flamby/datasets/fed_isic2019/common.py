import torch

from flamby.datasets.fed_isic2019.dataset import FedIsic2019

NUM_CLIENTS = 6
BATCH_SIZE = 32
BATCH_SIZE_POOLED = 32
NUM_EPOCHS_POOLED = 10
LR = 0.001
Optimizer = torch.optim.Adam
dropout = 0.1
FedClass = FedIsic2019


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    # TODO find out true number
    return (18413 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
