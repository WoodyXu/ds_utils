"""
RoI(Regions of Interest) Pooling for Fast R-CNN
use tensorflow image dim ordering (width, height, channel)
"""
import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer

class RoiPooling(Layer):
    """
    Input shape:
        list of two 4D tensors [images, rois]
        images shape: (batch, width, height, channel)
        rois shape: (batch, num_rois, 4)
                    the fours are (x, y, w, h)
    Output shape:
        (batch, num_rois, pool_size, pool_size, channel)
    """
    def __init__(self, pool_size, num_rois, **kwargs):
        """
        Parameters:
            pool_size: the output size of pooling layer 
            num_rois: number of regions of interest in one image
        """
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Define weights
        """
        self.num_channels = input_shape[0][3]
        super(RoiPooling, self).build(input_shape)

    def compute_output_shape(slf, input_shape):
        """
        calculate output shape
        """
        return (None, self.num_rois, self.pool_size, self.pool_size, self.num_channels)

    def call(self, x, mask=None):
        """
        layer logic
        Parameters:
            x: input tensor
        """
        assert len(x) == 2
        imgs, rois = x

        outputs = []

        for index in range(self.num_rois):
            x = rois[0, index, 0]
            y = rois[0, index, 1]
            w = rois[0, index, 2]
            h = rois[0, index, 3]

            x = K.cast(x, "int32")
            y = K.cast(y, "int32")
            w = K.cast(w, "int32")
            h = K.cast(h, "int32")

            result = tf.image.resize_images(imgs[0, x: x + w, y: y + h, :], (self.pool_size, self.pool_size))
            outputs.append(result)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.num_channels))
        return final_output
