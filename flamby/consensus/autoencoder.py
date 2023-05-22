import torch
import numpy as np
from scipy.special import softmax
from monai.networks.nets import AutoEncoder
from torch import optim, nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dummy_metric(place_holder1, place_holder2):
    return 1


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


def ae_combiner(task, preds_concs, scores):
    aggregated_preds = {k: [] for k in preds_concs.keys()}
    for test_set in preds_concs.keys():
        for element_index in range(len(preds_concs[test_set])):
            w = np.stack(scores[test_set][element_index], axis=0)
            w = softmax(1 / w, axis=0)
            p = np.stack(preds_concs[test_set][element_index], axis=0)
            tmp = np.average(p, axis=0, weights=w)
            aggregated_preds[test_set].append(tmp)
    return aggregated_preds
