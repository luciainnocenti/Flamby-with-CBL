import pickle

import numpy as np
from scipy.special import softmax
from collections import Counter


def average_combiner(task, preds_concs):
    mv_predictions = {k: [] for k in preds_concs.keys()}
    for test_set in list(preds_concs.keys()):
        for prediction_element in preds_concs[test_set]:
            mv_predictions[test_set].append(np.mean(prediction_element, axis=0))
    return mv_predictions
