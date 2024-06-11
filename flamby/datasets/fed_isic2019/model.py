import random

import albumentations
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from flamby.datasets.fed_isic2019 import FedIsic2019


class Baseline(nn.Module):
    """Baseline model
    We use the EfficientNets architecture that many participants in the ISIC
    competition have identified to work best.
    See here the [reference paper](https://arxiv.org/abs/1905.11946)
    Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
    [pytorch reimplementation of EfficientNets]
    (https://github.com/lukemelas/EfficientNet-PyTorch).
    """

    def __init__(self, dropout, pretrained=True, arch_name="efficientnet-b1"):
        super(Baseline, self).__init__()
        self.pretrained = pretrained
        self.overide_params = {'dropout_rate': dropout, 'include_top': True}
        self.base_model = (
            EfficientNet.from_pretrained(arch_name, **self.overide_params)
            if pretrained
            else EfficientNet.from_name(arch_name, **self.overide_params)
        )
        # self.base_model=torchvision.models.efficientnet_v2_s(pretrained=pretrained)
        nftrs = self.base_model._fc.in_features
        print("Number of features output by EfficientNet", nftrs)
        self.base_model._fc = nn.Linear(nftrs, 256)
        self.output = nn.Sequential(nn.LeakyReLU(),
                                    nn.Dropout(p=dropout),
                                    nn.Linear(256, 8))
        self.relu = nn.ReLU()
        self._dropout = nn.Dropout(p=dropout)
        
    def forward(self, image):
        out = self.base_model(image)

        out = self._dropout(out)
        out = self.relu(out)
        out = self._final_fc(out)
        return out
