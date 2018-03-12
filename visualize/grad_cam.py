"""
Grad-CAM: Gradient weighted Class Activation Map
Gradient-weighted Class Activation Mapping (Grad-CAM), 
uses the gradients of any target concept (say logits for ‘dog’ or even a caption), 
flowing into the final convolutional layer to produce a coarse localization map 
highlighting the important regions in the image for predicting the concept.
"""

import sys

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import keras.backend as K

import numpy as np
import cv2

# use vgg16 as an example, the default input size of which is 224*224
# other pre trained models are also available
model = VGG16(weights="imagenet")
img_path = sys.argv[1]
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# normalize
x = preprocess_input(x)

pred = model.predict(x)[0]
class_idx = np.argmax(pred)
# the target tensor
class_output_tensor = model.output[:, class_idx]
# 14*14*512
last_conv_layer_tensor = model.get_layer("block5_conv3")

grads_tensor = K.gradients(class_output_tensor, last_conv_layer_tensor.output)[0]
pooled_grads_tensor = K.mean(grads_tensor, axis=(0, 1, 2))
iterate = K.function([model.input], [ pooled_grads_tensor, last_conv_layer_tensor.output[0] ])

pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# drop channels
heatmap = np.mean(conv_layer_output_value, axis=-1)
print "Hint: the shape of the heatmap: {}".format(heatmap.shape)
# activate: relu
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# show
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], imag.shape[0]))
heatmap = np.uint8(heatmap * 255)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
heatmap = cv2.addWeights(img, 0.6, heatmap, 0.4, 0)
cv2.imshow("Original", img)
cv2.imshow("GradCAM", heatmap)
