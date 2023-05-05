from scipy.special import softmax


def normalizer(to_norm: bool):
    if to_norm:
        return lambda x: x
    else:
        return softmax


def create_data_structure(y_pred_dicts, y_true_dicts, to_norm=False):
    clients = list(y_pred_dicts.keys())
    test_sets = list(y_pred_dicts[clients[0]].keys())
    preds_concs = {}
    m = normalizer(to_norm=to_norm)
    for test_set in test_sets:
        preds_concs[test_set] = []
        for i in range(len(y_pred_dicts[clients[0]][test_set])):
            preds_concs[test_set].append([m(y_pred_dicts[local][test_set][i]) for local in y_pred_dicts.keys()])

    gt_concs = {}
    client = clients[0]
    for test_set in test_sets:
        gt_concs[test_set] = []
        for i in range(len(y_true_dicts['Local 0'][test_set])):
            gt_concs[test_set].append([y_true_dicts[client][test_set][i]])

    return preds_concs, gt_concs
