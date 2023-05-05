import numpy as np
import SimpleITK as sitk


def staple_combiner(task, preds_concs):
    assert task == "segmentation", "Staple is only applicable in segmentation problems"
    aggregated_preds = {k: [] for k in preds_concs.keys()}
    for test_set in list(preds_concs.keys()):
        for prediction_element in preds_concs[test_set]:
            images = []
            backgrounds = []
            for client_prediction in prediction_element:
                tmp = (client_prediction > 0.5)[1, :].astype(np.int16)
                tmp = sitk.GetImageFromArray(tmp)
                images.append(tmp)
                tmp = (client_prediction > 0.5)[0, :].astype(np.int16)
                tmp = sitk.GetImageFromArray(tmp)
                backgrounds.append(tmp)
            tmp_img = sitk.STAPLE(images, 1.0)
            tmp_img = sitk.GetArrayFromImage(tmp_img > 0.5)
            tmp_bkg = sitk.STAPLE(backgrounds, 1.0)
            tmp_bkg = sitk.GetArrayFromImage(tmp_bkg > 0.5)
            aggregated_preds[test_set].append(np.stack([tmp_img, tmp_bkg]))
    return aggregated_preds
