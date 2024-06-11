import os

import torch
import numpy as np
from scipy.special import softmax
from monai.networks.nets import AutoEncoder
from torch import optim, nn
import sys
import copy

target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dummy_metric(place_holder1, place_holder2):
    return 1


def train_autoencoder_image(dataloader: torch.utils.data.DataLoader,
                            task: str,
                            dataset_name: str,
                            i: int,
                            epochs: int = 50,
                            modality: str = 'image'):
    if task == 'segmentation':
        model = AutoEncoder(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16,),
            strides=(2,),
            inter_channels=[8, 8, 8],
            inter_dilations=[1, 2, 4],
            num_inter_units=2)

    else:
        model = AutoEncoder(
            spatial_dims=2,
            in_channels=3,
            out_channels=3,
            channels=[16, 32, 64],
            strides=[2, 2, 2],
            inter_channels=[128, 64, 32],
            inter_dilations=[1, 1, 1],
            num_inter_units=2
        )
    if os.path.exists(f"ae_{dataset_name}_local_{i}_{modality}"):
        print("Loading trained autoencoder")
        model.load_state_dict(torch.load(f"ae_{dataset_name}_local_{i}_{modality}"))
    else:
        print(f"Training ae_{dataset_name}_local_{i}_{modality}")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        final_loss = 0

        for epoch in range(epochs):
            loss = 0
            for batch_features, batch_targets in dataloader:
                batch_input = batch_features if modality == 'image' else batch_targets[:, :1, :]
                optimizer.zero_grad()
                outputs = model(batch_input)
                train_loss = criterion(outputs, batch_input)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
            loss = loss / len(dataloader)
            final_loss += loss
        final_loss /= epochs
        print(f"ae for model {dataset_name} trained with loss {final_loss}")
        torch.save(model.state_dict(), f"ae_{dataset_name}_local_{i}_{modality}")
    return model


def train_autoencoder_tabular(dataloader: torch.utils.data.DataLoader,
                              dataset_name: str,
                              i: int,
                              epochs: int = 5):
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    input_shape = 39  # 13
    model = TabularAE(input_shape=input_shape)
    model.to(target_device)

    if os.path.exists(f"ae_{dataset_name}_local_{i}"):
        print("Loading trained autoencoder")
        model.load_state_dict(torch.load(f"ae_{dataset_name}_local_{i}"))
    else:
        # create an optimizer object
        # Adam optimizer with learning rate 1e-3
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # mean-squared error loss
        criterion = nn.BCEWithLogitsLoss()
        m = nn.Sigmoid()
        final_loss = 0
        for epoch in range(epochs):
            loss = 0
            for batch_features, _ in dataloader:
                batch_features = batch_features.type(torch.float32).to(target_device)
                optimizer.zero_grad()
                outputs = model(batch_features)
                outputs = m(outputs)
                train_loss = criterion(outputs, batch_features)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
            loss = loss / len(dataloader)
            final_loss += loss
        final_loss /= epochs
        print(f"ae for model {dataset_name} trained with loss {final_loss}")
        torch.save(model.state_dict(), f"ae_{dataset_name}_local_{i}")
    return model


class SimpleAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(in_features=kwargs['input_shape'],
                                 out_features=kwargs['input_shape'] // 2
                                 )
        self.decoder = nn.Linear(in_features=kwargs["input_shape"] // 2,
                                 out_features=kwargs["input_shape"]
                                 )

    def forward(self, features):
        features = features.to(self.encoder.weight.dtype)
        x = self.encoder(features)
        x = self.decoder(x)
        return x


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
        features = features.to(self.encoder_hidden_layer.weight.dtype)
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
        test_set_scores = scores[test_set]
        test_set_elements = preds_concs[test_set]
        for element_index in range(len(test_set_elements)):
            tmp = compute_ae_average(test_set_elements[element_index],
                                     test_set_scores[element_index])
            aggregated_preds[test_set].append(tmp)
    return aggregated_preds


def compute_ae_average(elements, scores):
    w = np.stack(scores, axis=0)
    p = np.stack(elements, axis=0)
    tmp = np.average(p, axis=0, weights=w)
    return tmp


def ae_weights_computation(models, test_dls, use_gpu, typology):
    epsilon = sys.float_info.epsilon
    results = {f'client_test_{i}': [] for i in range(len(test_dls))}
    criterion = nn.MSELoss()
    for i, test_dl in enumerate(test_dls):
        for j, (X, y) in enumerate(test_dl):
            X = X.to(target_device)
            y = y.to(target_device)

            if typology == "ae_data":
                input_ae = copy.deepcopy(X)
            else:
                input_ae = copy.deepcopy(y[:, :1, :])
            res = np.array([])
            for model in models.values():
                model.to(target_device)
                model.eval()
                output = model(input_ae)
                output = output.detach()
                s = criterion(output, input_ae).item()
                res = np.append(res, s)
            w = softmax(1 / (res + epsilon), axis=0)
            results[f'client_test_{i}'].append(w)
    return results
