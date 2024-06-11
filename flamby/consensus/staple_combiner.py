import numpy as np
import SimpleITK as sitk
import torch


def staple_combiner(task, preds_concs):
    assert task == "segmentation", "Staple is only applicable in segmentation problems"
    aggregated_preds = {k: [] for k in preds_concs.keys()}
    for test_set in list(preds_concs.keys()):
        for prediction_element in preds_concs[test_set]:
            tmp_img, tmp_bkg = compute_staple(prediction_element)
            aggregated_preds[test_set].append(np.stack([tmp_bkg, tmp_img]))
    return aggregated_preds


def compute_staple(prediction_element):
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
    return tmp_img, tmp_bkg


def compute_staple_multiclass(prediction_element, num_classes):
    seg_classes = {}
    for i in range(1, num_classes):
        images = []
        for client_prediction in prediction_element:
            tmp = (client_prediction == i)[0, 0, :].astype(np.int16)
            tmp_img = sitk.GetImageFromArray(tmp)
            images.append(tmp_img)
        seg_classes[f'class_{i}'] = images
    cs = []
    for i in range(1, num_classes):
        consensus_tmp = sitk.STAPLE(seg_classes[f'class_{i}'], 1)
        consensus_seg_class_np = sitk.GetArrayFromImage(consensus_tmp)
        consensus_seg_class_np = (consensus_seg_class_np > 0.5).astype(np.int16)
        # Add the modified consensus segmentation for the current class to the list
        cs.append(consensus_seg_class_np)
    final_segmentation = np.where(cs[0] == 0, 0, np.where(cs[1] == 0, 1, 2))
    return final_segmentation[None, :]
