import torch
import sys
import numpy as np
from scipy.special import softmax


def dummy_metric(place_holder1, place_holder2):
    return 1


def uncertainty_computation(models, evaluate_func, test_dls,
                            use_gpu, N=50):

    results = {f'Local {i}': {} for i in range(len(models))}
    for local, model in models.items():
        concatenations = None
        if use_gpu:
            model.cuda()
        for i in range(N):
            prediction = evaluate_func(model, test_dls, dummy_metric, return_pred=True, dropout=True)
            _, _, y_pred_dict = prediction
            if concatenations is None:
                concatenations = {t: [-1] * len(y_pred_dict[t]) for t in list(y_pred_dict.keys())}
                results[local] = {t: [-1] * len(y_pred_dict[t]) for t in list(y_pred_dict.keys())}

            for test_set in list(y_pred_dict.keys()):
                for counter, el in enumerate(y_pred_dict[test_set]):
                    if concatenations[test_set][counter] == -1:
                        concatenations[test_set][counter] = [el]
                    else:
                        concatenations[test_set][counter].append(el)

        for test_set in concatenations.keys():
            for i, el in enumerate(concatenations[test_set]):
                tmp1 = torch.as_tensor(el)
                std_tmp = torch.std(tmp1, 0)
                tmp_mean = torch.mean(std_tmp)
                results[local][test_set][i] = tmp_mean

    weights_concs = {}
    for test_set in results['Local 0']:
        weights_concs[test_set] = []
        for i in range(len(results['Local 0'][test_set])):
            weights_concs[test_set].append([results[local][test_set][i] for local in results.keys()])
    return weights_concs


def uncertainty_combiner(task, preds_concs, scores):
    aggregated_preds = {k: [] for k in preds_concs.keys()}
    for test_set in preds_concs.keys():
        test_set_scores = scores[test_set]
        test_set_elements = preds_concs[test_set]
        for element_index in range(len(test_set_elements)):
            tmp = compute_unc_average(test_set_elements[element_index], test_set_scores[element_index])
            aggregated_preds[test_set].append(tmp)
    return aggregated_preds


def compute_unc_average(elements, scores):
    epsilon = sys.float_info.epsilon
    w = np.stack(scores, axis=0)
    w = softmax(1 / (w + epsilon), axis=0)
    p = np.stack(elements, axis=0)
    tmp = np.average(p, axis=0, weights=w)
    return tmp
