# -*- encoding: utf-8 -*-

"""
Author: Woody
Description: This is the Residual Net Representation module for 1-D Input which is full of MLPs
"""

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import keras
from keras.layers import Input, Dense, add
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

class ResNet(object):
    def _bn_relu(input):
        """
        Parameters:
            input: input
        Returns:
            BN + PRELU
        """
        norm = BatchNormalization()(input)
        return PReLU()(norm)

    def __mlp_bn_relu(hiddens):
        """
        customize layer
        """

        def f(input):
            mlp = Dense(hiddens, kernel_regularizers=keras.regularizers.l2(0.0001))(input)
            return self._bn_relu(mlp)
        return f

    def _bn_relu_mlp(hiddens):
        """
        customize layer
        """
        def f(input):
            act = self._bn_relu(input)
            return Dense(hiddens, kernel_regularizers=keras.regularizers.l2(0.0001))(act)
        return f

    def _shortcut(input, residual):
        return add([input, residual])

    def _residual_block(hiddens, repeat, is_first_layer=False):
        """
        customize layer
        """
        def f(input):
            mlp = input
            for index in range(repeat):
                input = mlp
                if is_first_layer and index == 0:
                    mlp = Dense(hiddens, kernel_regularizers=keras.regularizers.l2(0.0001))(input)
                else:
                    mlp = self._bn_relu_mlp(hiddens)(input)

                residual = self._bn_relu_mlp(hiddens)(input)
                mlp = self._shortcut(input, mlp)

            return mlp
        return f

    @staticmethod
    def build(input_layer, hiddens, num_output, repetitions):
        block = self._mlp_bn_relu(hiddens)(input_layer)
        for index, value in enumerate(repetitions):
            block = self._residual_block(hiddens, value, index == 0)(block)
        block = self._bn_relu(block)
        return block

    @staticmethod
    def build_resnet_18(input_layer, hiddens, num_output):
        return ResNet.build(input_layer, hiddens, num_output, [2, 2, 2, 2])
