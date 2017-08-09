# -*- encoding: utf-8 -*-

"""
Author: Woody
Description: This module is for binary model evaluation, include auc, ks, precision, recall,
accuracy and optimal cut point.
"""

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics

import predict_evaluate

def model_evaluate(truth, pred):
    """
    Parameters:
        truth: a list of ground truth, 1 or 0
        pred: a list of guess probabilities, range [0, 1]
    Returns:
        auc, ks, optimal cut point, accuracy, precision and recall
    Raises:
        ValueError if both lengths of input are not equal
    """
    if len(truth) != len(pred):
        raise ValueError("Lengths of truth and guesst must be equal!")

    truth = np.asarray(truth).astype(int)
    pred = np.asarray(pred).astype(float)

    # auc
    fpr, tpr, thresholds = metrics.roc_curve(truth, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # ks
    data = pd.DataFrame({"pos": truth, "pred": pred})
    data["neg"] = 1 - data["pos"]
    data["bucket"] = pd.qcut(data["pred"], 1000)
    print data
    grouped = data.groupby('bucket', as_index=False)

    agg1 = pd.DataFrame()
    agg1["min_pred"] = grouped.min()["pred"]
    agg1["max_pred"] = grouped.max()["pred"]
    agg1["pos_num"] = grouped.sum()["pos"]
    agg1["neg_num"] = grouped.sum()["neg"]
    agg1["total"] = agg1["pos_num"] + agg1["neg_num"]
    print agg1

    agg2 = (agg1.sort_values(by='min_pred')).reset_index(drop=True)
    agg2["ks"] = np.abs((agg2["neg_num"] * 1.0 / agg2["neg_num"].sum()).cumsum() - \
            (agg2["pos_num"] * 1.0 / agg2["pos_num"].sum()).cumsum()) * 100

    print agg2

    ks = agg2["ks"].max()
    opt_index = agg2["ks"].argmax()
    opt_min, opt_max = agg2["min_pred"].iloc[opt_index], agg2["max_pred"].iloc[opt_index]
    opt_cut = (opt_min + opt_max) / 2.0

    tp = np.sum(np.logical_and(pred > opt_cut, truth == 1))
    fp = np.sum(np.logical_and(pred > opt_cut, truth == 0))
    fn = np.sum(np.logical_and(pred < opt_cut, truth == 1))
    tn = np.sum(np.logical_and(pred < opt_cut, truth == 0))

    accuracy = (tp + tn) * 1.0 / (tp + fp + fn + tn)
    precision = tp * 1.0 / (tp + fp + 0.000001)
    recall = tp * 1.0 / (tp + fn + 0.0000001)

    return auc, ks, opt_cut, accuracy, precision, recall

if __name__ == "__main__":
    preds = np.random.random(10000)
    truths = np.random.randint(2, size=10000)

    print model_evaluate(truths, preds)
