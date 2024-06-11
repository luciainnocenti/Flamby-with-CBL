import pickle

import numpy as np
import torch
from scipy.special import softmax
from collections import Counter


def majority_voting_combiner(task, preds_concs):
    if task == "segmentation":
        return majority_voting_segmentation(preds_concs)
    else:
        return majority_voting_classification(preds_concs)


def majority_voting_classification(preds_concs):
    class_predictions = {k: [] for k in preds_concs.keys()}
    num_classes = 0
    for test_set in list(preds_concs.keys()):
        for prediction_element in preds_concs[test_set]:
            tmp = []
            for client_prediction in prediction_element:
                num_classes = len(client_prediction)
                if len(client_prediction) == 1:
                    tmp.append(int(client_prediction >= 0.5))
                else:
                    tmp.append(np.argmax(client_prediction))
            class_predictions[test_set].append(tmp)

    mv_predictions = {k: [] for k in preds_concs.keys()}
    for test_set in list(preds_concs.keys()):
        for classes in class_predictions[test_set]:
            data = Counter(classes)
            xx = max(classes, key=data.get)
            placeholder = [0] * (num_classes+1)
            placeholder[xx] = 1
            mv_predictions[test_set].append(placeholder)
    return mv_predictions


def majority_voting_segmentation(preds_concs):
    aggregated_preds = {k: [] for k in preds_concs.keys()}
    for test_set in list(preds_concs.keys()):
        for prediction_element in preds_concs[test_set]:
            tmp = compute_majority(prediction_element)
            aggregated_preds[test_set].append(tmp)
    return aggregated_preds


def compute_majority(prediction_element):
    sums = np.zeros(prediction_element[0].shape)
    for i, client_prediction in enumerate(prediction_element):
        sums = [a + b for a, b in zip(sums, client_prediction)]
    tmp = np.divide(sums, len(prediction_element))
    tmp = np.stack(tmp)
    tmp = (tmp > 0.5).astype(np.int_)
    return tmp


def compute_majority_multiclass(prediction_element):
    stacked = torch.stack([torch.Tensor(x) for x in prediction_element])
    majority_voting, _ = torch.mode(stacked, dim=0)
    return majority_voting
