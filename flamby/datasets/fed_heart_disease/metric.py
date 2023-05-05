import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score


def metric(y_true, y_pred):
    y_true = y_true.astype("uint8")
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        # return roc_auc_score(y_true, y_pred)
        # proposed modification in order to get a metric that calcs on center 2
        # (y=1 only on that center)
        if not (np.sum(y_pred, axis=1) == 1).all():
            y_pred = softmax(y_pred, axis=1)
        m = np.argmax(y_pred, axis=1)
        return (m == y_true[:, 0]).mean()
    except ValueError:
        return np.nan