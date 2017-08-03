# -*- encoding: utf-8 -*-

from sklearn import metrics

import keras

class AUC_Callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get("val_loss", -1))
        y_pred = self.model.predict(self.validation_data[0])
        fpr, tpr, thresholds = metrics.roc_curve(self.validation_data[1], y_pred,
                                                 pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print "At Epoch %d, the current auc is: %f" % (epoch + 1, auc)