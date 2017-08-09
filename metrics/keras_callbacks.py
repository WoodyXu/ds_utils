# -*- encoding: utf-8 -*-

"""
Author: Woody
Description: This is the keras callback for binary classification evaluation
"""

from sklearn import metrics
import keras

import model_evaluate

class Metrics_Callback(keras.callbacks.Callback):
    """Print auc at the end of each epoch, only supporting binary classification.
    """
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get("val_loss", -1))
        y_pred = self.model.predict(self.validation_data[0])
        fpr, tpr, thresholds = metrics.roc_curve(self.validation_data[1], y_pred,
                                                 pos_label=1)

        auc, ks, opt_cut, accuracy, precision, recall = \
        model_evaluate.model_evaluate(self.validation_data[1], y_pred)
        print "\nAt Epoch %d, the current metrics information as follows:\nauc is: %.5f\nks is: %.5f\nopt cut \
        is: %.5f\naccuracy is: %.5f\nprecision is: %.5f\nrecall is: %.5f\n" % (epoch + 1, auc, ks,
                opt_cut, accuracy, precision, recall)
