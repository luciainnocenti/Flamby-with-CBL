import pickle

import numpy as np
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
            placeholder = [0]*num_classes
            placeholder[xx] = 1
            mv_predictions[test_set].append(placeholder)
    return mv_predictions


def majority_voting_segmentation(preds_concs):
    aggregated_preds = {k: [] for k in preds_concs.keys()}
    for test_set in list(preds_concs.keys()):
        for prediction_element in preds_concs[test_set]:
            sums = np.zeros(len(prediction_element[0]))
            for client_prediction in prediction_element:
                sums = [a + b for a, b in zip(sums, client_prediction)]
            tmp = np.divide(sums, len(prediction_element))
            tmp = (tmp > 0.5).astype(np.int_)
            aggregated_preds[test_set].append(tmp)
    return aggregated_preds

