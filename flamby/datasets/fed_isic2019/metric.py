import numpy as np
from sklearn import metrics


def metric(y_true, logits):
    y_true = y_true.reshape(-1)
    try:
        preds = np.argmax(logits, axis=1)
    except:
        preds = logits
    return metrics.balanced_accuracy_score(y_true, preds)
