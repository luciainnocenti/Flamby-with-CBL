import torch
import numpy as np
from scipy.special import softmax


def dummy_metric(place_holder1, place_holder2):
    return 1


def uncertainty_computation(models, evaluate_func, test_dls,
                            use_gpu, N=50, project_dir=""):
    concatenations = {f'Local {i}': None for i in range(len(models))}
    results = {f'Local {i}': None for i in range(len(models))}
    for local, model in models.items():
        if use_gpu:
            model.cuda()
        for i in range(N):
            prediction = evaluate_func(model, test_dls, dummy_metric, return_pred=True, dropout=True)
            _, _, y_pred_dict = prediction
            if concatenations[local] is None:
                concatenations[local] = {t: [-1] * len(y_pred_dict[t]) for t in list(y_pred_dict.keys())}
                results[local] = {t: [-1] * len(y_pred_dict[t]) for t in list(y_pred_dict.keys())}

            for test_set in list(y_pred_dict.keys()):
                for counter, el in enumerate(y_pred_dict[test_set]):
                    if concatenations[local][test_set][counter] == -1:
                        concatenations[local][test_set][counter] = [el]
                    else:
                        concatenations[local][test_set][counter].append(el)

    import pickle
    with open(f'{project_dir}/concatenations.obj', 'wb') as outp:
        pickle.dump(concatenations, outp)

    for local in list(concatenations.keys()):
        for test_set in concatenations[local].keys():
            for i, el in enumerate(concatenations[local][test_set]):
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


def uncertainty_combiner(task, preds_concs, scores, project_dir):
    aggregated_preds = {k: [] for k in preds_concs.keys()}
    for test_set in preds_concs.keys():
        for element_index in range(len(preds_concs[test_set])):
            w = np.stack(scores[test_set][element_index], axis=0)
            w = softmax(1 / w, axis=0)
            p = np.stack(preds_concs[test_set][element_index], axis=0)
            tmp = np.average(p, axis=0, weights=w)
            aggregated_preds[test_set].append(tmp)
    return aggregated_preds
