import numpy as np


def max_voting_combiner(task, preds_concs):
    if task == "segmentation":
        return majority_voting_segmentation(preds_concs)
    else:
        return majority_voting_classification(preds_concs)


def majority_voting_classification(preds_concs):
    class_predictions = {k: [] for k in preds_concs.keys()}
    for test_set in preds_concs.keys():
        for i, prediction_element in enumerate(preds_concs[test_set]):
            num_classes = len(prediction_element[0])
            tmp = np.where(prediction_element == np.max(np.stack(prediction_element, axis=0)))
            placeholder = [0]*num_classes
            placeholder[tmp[-1][0]] = 1
            class_predictions[test_set].append(placeholder)
    return class_predictions



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

