"""
calculate the mAP (mean average precision) for object detection task
Normally we calculate mAP@iou_threshold
"""
import numpy as np
from sklearn.metrics import average_precision_score

from ..vision import iou

def calc_map(iou_threshold, pred, truth):
    """
    Calculate the mAP@iou_threshold, all the coordinates are in the original image space
    Parameters:
        iou_threshold: only greater than this value it can count
        pred: prediction for one image
            [roi1, roi2, roi3, ...]
            roi: {"x1": 1, "x2": 2, "y1": 1, "y2": 2, "class": "cat", "prob": 0.8}
        truth: ground truth for one image
            [bbox1, bbox2, bbox3, ...]
            bbox: {"x1": 1, "x2": 2, "y1": 1, "y2": 2, "class": "cat"}
    Returns:
        mAP@iou_threshold
    """
    result_trues, result_probs = {}, {}

    # we use this flag to identify whether there is a roi ever matched for each bounding box
    for bbox in truth:
        bbox["bbox_matched"] = False

    pred_probs = np.array([item["prob"] for item in pred])
    preds_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for pred_index in preds_sorted_by_prob:
        pred_item = pred[pred_index]
        pred_class = pred_item["class"]
        pred_prob = pred_item["prob"]
        pred_x1, pred_x2, pred_y1, pred_y2 = pred_item["x1"], pred_item["x2"], pred_item["y1"], pred_item["y2"]

        if pred_class not in result_probs:
            result_probs[pred_class] = []
            result_trues[pred_class] = []

        result_probs[pred_class].append(pred_prob)
        # identify whether True Positive or False Positive
        found_match = False

        for bbox in truth:
            truth_class = bbox["class"]
            truth_x1, truth_x2, truth_y1, truth_y2 = bbox["x1"], bbox["x2"], bbox["y1"], bbox["y2"]

            # if pred and truth are not the same
            if truth_class != pred_class:
                continue
            # if the current bounding box is occupied by anthoer roi
            # this makes us not to repeat to count true positives 
            if bbox["bbox_matched"] is True:
                continue

            current_iou = iou.iou((pred_x1, pred_y1, pred_x2, pred_y2),
                                  (truth_x1, truth_y1, truth_x2, truth_y2))
            if current_ious >= iou_threshold:
                found_match = True
                bbox["bbox_matched"] = True
                break
            else:
                continue

        result_truth[pred_class].append(int(found_match))

    # we also can't miss False Negative
    for bbox in truth:
        if bbox["bbox_matched"] is False:
            if bbox["class"] not in result_probs:
                result_probs[bbox["class"]] = []
                result_truth[bbox["class"]] = []
            result_probs[bbox["class"]].append(0)
            result_truth[bbox["class"]].append(1)

    # ok now we start to calculate the metrics
    all_aps = []
    for class_item in result_truth:
        ap = average_precision_score(result_truth[class_item], result_probs[class_item])
        all_aps.append(ap)
    return np.mean(all_aps)
