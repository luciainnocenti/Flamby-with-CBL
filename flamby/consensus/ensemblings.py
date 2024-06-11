import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from flamby.consensus.majority_voting import majority_voting_combiner, compute_majority_multiclass
from flamby.consensus.max_majority_voting import max_voting_combiner
from flamby.consensus.staple_combiner import staple_combiner, compute_staple, compute_staple_multiclass
from flamby.consensus.uncertainty import uncertainty_combiner, compute_unc_average
from flamby.consensus.autoencoder import ae_combiner, compute_ae_average
from flamby.consensus.average import average_combiner
from flamby.datasets.fed_kits19.metric import softmax_helper


def ensemble_perf_from_predictions(
        strategy, y_true_dicts, y_pred_dicts,
        num_clients, metric, weights, task, num_clients_test=None
):
    print("Computing ensemble performance from predictions")
    ensemble_preds = {}
    ensemble_true = {}
    if num_clients_test is None:
        num_clients_test = num_clients
    for testset_idx in range(num_clients_test):
        # Small safety net
        for model_idx in range(1, num_clients):
            assert (
                    y_true_dicts[f"Local {0}"][f"client_test_{testset_idx}"]
                    == y_true_dicts[f"Local {model_idx}"][f"client_test_{testset_idx}"]
            ).all(), "Models in the ensemble have different ground truths"

        # Since they are all the same we use the first one
        # for this specific tests as the ground truth
        ensemble_true[f"client_test_{testset_idx}"] = y_true_dicts["Local 0"][f"client_test_{testset_idx}"]

    # Retrieve the predictions from the ensembling function
    if strategy == "mav":
        ensemble_pred = majority_voting_combiner(task=task, preds_concs=y_pred_dicts)
    elif strategy == 'staple':
        ensemble_pred = staple_combiner(task=task, preds_concs=y_pred_dicts)
    elif strategy == 'uncertainty':
        ensemble_pred = uncertainty_combiner(task=task, preds_concs=y_pred_dicts,
                                             scores=weights)
    elif strategy == "maxv":
        ensemble_pred = max_voting_combiner(task=task, preds_concs=y_pred_dicts)
    elif strategy in ['ae_data', 'ae_label']:
        ensemble_pred = ae_combiner(task=task, preds_concs=y_pred_dicts, scores=weights)
    elif strategy == 'average':
        ensemble_pred = average_combiner(task=task, preds_concs=y_pred_dicts)
    else:
        return -1

    for testset_idx in range(num_clients_test):
        test_set = f"client_test_{testset_idx}"
        ensemble_preds[f"client_test_{testset_idx}"] = metric(ensemble_true[test_set], np.array(ensemble_pred[test_set]))
    return ensemble_preds


def ensemble_perf_from_models(
        strategy, dls, models, num_clients, metric, weights, task, num_clients_test=None
):
    print("Computing ensemble performance from models for ", strategy)
    ensemble_preds = {}
    if num_clients_test is None:
        num_clients_test = num_clients
    with torch.inference_mode():
        for i in tqdm(range(len(dls))):
            test_dataloader_iterator = iter(dls[i])
            ensemble_preds[f"client_test_{i}"] = []
            perf_concs = []
            cnt = 0
            for (X, y) in test_dataloader_iterator:
                preds_concs = {}
                for local, model in models.items():
                    if torch.cuda.is_available():
                        X = X.cuda()
                        model.cuda()
                    y_pred = model(X).detach().cpu()
                    pred_softmax = softmax_helper(y_pred)
                    preds_concs[local] = pred_softmax.argmax(1).unsqueeze(dim=0).numpy()
                num_classes = 3  # int(preds_concs['Local 0'].max()) + 1
                if strategy == "mav":
                    ensembles = compute_majority_multiclass(list(preds_concs.values()))
                elif strategy == 'staple':
                    ensembles = compute_staple_multiclass(list(preds_concs.values()), num_classes)
                elif strategy == 'uncertainty':
                    scores = weights[f"client_test_{i}"][cnt:(cnt + dls[i].batch_size)]
                    scores = np.reshape(np.array(scores), [np.array(scores).size])
                    ensembles = compute_unc_average(list(preds_concs.values()), scores)
                elif strategy == 'ae_data':
                    scores = weights[f"client_test_{i}"][cnt:(cnt + dls[i].batch_size)]
                    scores = np.reshape(np.array(scores), [np.array(scores).size])
                    ensembles = compute_ae_average(list(preds_concs.values()), scores)
                elif strategy == 'average':
                    ensembles = np.mean(list(preds_concs.values()), axis=0)
                perfs_tmp = metric(y.detach()[0, :], torch.tensor(ensembles))
                perf_concs.append(perfs_tmp.item())
            ensemble_preds[f'client_test_{i}'] = np.mean(perf_concs)
    return ensemble_preds
