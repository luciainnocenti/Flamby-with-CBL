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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_ds_task = {
    'fed_camelyon16': '',
    'fed_heart_disease': 'tab_classification',
    'fed_isic2019': 'img_classification',
    'fed_ixi': 'segmentation',
    'fed_kits19': 'segmentation',
    'fed_lidc_idri': 'segmentation',
    'fed_tcga_brca': 'tab_regression'
}
map_task_ensembles = {
    'tab_classification': ["mav", "uncertainty", "maxv", "average", 'ae_data'],
    'img_classification': ["ae_data", "mav", "uncertainty", "average"],
    'segmentation': [ "average", "mav", "ae_data"], #"staple", "uncertainty",
    'tab_regression': ["ae_data", "uncertainty", "average"]
}


def get_task(dataset_name: str):
    assert dataset_name in list(map_ds_task.keys()), f'{dataset_name} not in available datasets'
    return map_ds_task[dataset_name]


def get_ensemble_methods(task: str):
    assert task in list(map_task_ensembles.keys()), f'task {task} not in list'
    return map_task_ensembles[task]

