import SimpleITK as sitk
from numpy import average
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from monai.networks.nets import AutoEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_ds_task = {
    'fed_camelyon16': '',
    'fed_heart_disease': 'tab_classification',
    'fed_isic2019': 'img_classification',
    'fed_ixi': 'segmentation',
    'fed_kits19': 'segmentation',
    'fed_lidc_idri': 'segmentation',
    'fed_tcga_brca': 'tab_classification'
}
map_task_ensembles = {
    'tab_classification': ["mav", "uncertainty", "maxv"],  # , 'ae'],
    'img_classification': ["mav", "uncertainty", "maxv"],  # , 'ae'],
    'segmentation': ["mav", "staple", "uncertainty"],  # , 'ae']
}


def get_task(dataset_name: str):
    assert dataset_name in list(map_ds_task.keys()), f'{dataset_name} not in available datasets'
    return map_ds_task[dataset_name]


def get_ensemble_methods(task: str):
    assert task in list(map_task_ensembles.keys()), f'task {task} not in list'
    return map_task_ensembles[task]


def train_autoencoder_image(dataloader: torch.utils.data.DataLoader, epochs: int = 50, modality: str = 'image'):
    model = AutoEncoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16,),
        strides=(2,),
        inter_channels=[8, 8, 8],
        inter_dilations=[1, 2, 4],
        num_inter_units=2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    final_loss = 0

    for epoch in range(epochs):
        loss = 0
        for batch_features, batch_targets in dataloader:
            batch_input = batch_features if modality == 'image' else batch_targets
            optimizer.zero_grad()
            outputs = model(batch_input)
            train_loss = criterion(outputs, batch_input)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(dataloader)
        final_loss += loss
    final_loss /= epochs
    return model, final_loss


def train_autoencoder_tabular(dataloader: torch.utils.data.DataLoader, epochs: int = 50):
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    dataiter = iter(dataloader)
    input_shape = dataiter.next().shape[-1]
    model = TabularAE(input_shape=input_shape).to(device)
    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()
    final_loss = 0
    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = criterion(outputs, batch_features)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(dataloader)
        final_loss += loss
    final_loss /= epochs
    return model, final_loss


class TabularAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=32
        )
        self.encoder_output_layer = nn.Linear(
            in_features=32, out_features=32
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=32, out_features=32
        )
        self.decoder_output_layer = nn.Linear(
            in_features=32, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

