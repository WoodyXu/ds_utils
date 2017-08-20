# -*- encoding: utf-8 -*-

"""
Author: Woody
Description: This module is for plotting model evaluations, including ROC Curve, KS Curve and PR Curve.
"""

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics

import model_evaluate as eva

def plot_metrics(truth, pred, threshold=None):
    """
    Parameters:
        truth: a list of ground truth, 1 or 0
        pred: a list of guess probabilities, range [0, 1]
        threshold: the optimal cut, default None
    Returns:
        No returns, just plot some plots, including ROC Curve, KS Curve and PR Curve
    Raises:
        ValueError if both lengths of inputs are not equal
    """
    if len(truth) != len(pred):
        raise ValueError("Lengths of truth and guess must be equal!")

    truth = np.asarray(truth).astype(int)
    pred = np.asarray(pred).astype(float)

    valid_indexes = np.logical_or(truth == 0, truth == 1)
    truth = truth[valid_indexes]
    pred = pred[valid_indexes]

    auc, ks, opt, acc, prec, rec = eva.model_evaluate(truth, pred)

    pos_data = pred[truth == 1]
    neg_data = pred[truth == 0]

    # KS Curve
    plt.clf()
    plt.hist(x=pos_data, bins=1000, range=(0.0, 1.0), color="green",
            label="Positive", normed=True, cumulative=True, histtype="step")
    plt.hist(x=neg_data, bins=1000, range=(0.0, 1.0), color="blue",
            label="Negative", normed=True, cumulative=True, histtype="step")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title("KS Curve")
    plt.xlabel("Predictions")
    plt.ylabel("Cumulative Percentage")
    plt.grid()
    plt.legend(loc="upper left")

    ax = plt.gca()
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:.2f}%'.format(x * 100) for x in vals])

    pos_cum = sum(pos_data <= opt) * 1.0 / len(pos_data)
    neg_cum = sum(neg_data <= opt) * 1.0 / len(neg_data)
    plt.plot([opt, opt], [pos_cum, neg_cum], color="red", linestyle="--")
    plt.text(opt + 0.012, opt + 0.012, "KS={:.2f}".format(ks), fontsize=12)

    plt.savefig("./ks_curve.png", dpi=150)

    # ROC Curve
    fpr, tpr, _ = metrics.roc_curve(truth, pred, pos_label=1)
    plt.clf()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.grid()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Ratio")
    plt.ylabel("True Positive Ratio")

    plt.text(0.512, 0.512, "AUC={:.2f}".format(auc), fontsize=12)

    plt.savefig("./roc_curve.png", dpi=150)

    # Precision-Recall Curve 
    precision, recall, _ = metrics.precision_recall_curve(truth, pred, pos_label=1) 
    plt.clf()
    plt.plot(recall, precision)
    plt.grid()
    plt.title("PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig("./pr_curve.png", dpi=150)


if __name__ == "__main__":
    preds = np.random.random(10000)
    truths = np.random.randint(2, size=10000)
    plot_metrics(truths, preds)
