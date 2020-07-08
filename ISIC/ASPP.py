import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print("--FF")
from keras.layers import Concatenate
slim = tf.contrib.slim
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Activation, add, multiply, Lambda
from keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from keras.layers.normalization import BatchNormalization

kinit = 'glorot_normal'

def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img
#def ASPP(inputs, out_puts, kernel_size, padding, dilation):

def ASPP(input, out_channel, name):
    aspp_list = []
    x = Conv2D(out_channel, (1, 1), strides=(1, 1), kernel_initializer=kinit, padding="same", dilation_rate=(1, 1), name='as_conv1_1')(input)
    x = BatchNormalization(name='as_conv1_1_bn')(x)
    x1= Activation('relu', name='as_conv1_1_act')(x)

    x = Conv2D(out_channel, (3, 3), strides=(1, 1),  kernel_initializer=kinit, padding="same", dilation_rate=(6, 6), name='as_conv2_1')(x1)
    x = BatchNormalization(name='as_conv2_1_bn')(x)
    x2 = Activation('relu', name='as_conv2_1_act')(x)

    x = Conv2D(out_channel, (3, 3), strides=(1, 1),  kernel_initializer=kinit, padding="same", dilation_rate=(12, 12), name='as_conv3_1')(x2)
    x = BatchNormalization(name='as_conv3_1_bn')(x)
    x3 = Activation('relu', name='as_conv3_1_act')(x)

    x = Conv2D(out_channel, (3, 3), strides=(1, 1),  kernel_initializer=kinit, padding="same", dilation_rate=(18, 18), name='as_conv4_1')(x3)
    x = BatchNormalization(name='as_conv4_1_bn')(x)
    x4 = Activation('relu', name='as_conv4_1_act')(x)

    x = AveragePooling2D((1, 1))(x4)
    x = Conv2D(out_channel, (1, 1), strides=(1, 1),  kernel_initializer=kinit, padding="same", name='as_conv5_1')(x)
    x = BatchNormalization(name='as_conv5_1_bn')(x)
    x5 = Activation('relu', name='as_conv5_1_act')(x)

    x = Concatenate(axis=3)([x1, x2, x3, x4, x5])

    x = Conv2D(out_channel, (1, 1), strides=(1, 1), kernel_initializer=kinit, padding="same", name='as_conv6_1')(x)
    x = BatchNormalization(name='as_conv6_1_bn')(x)
    x = Activation('relu', name='as_conv6_1_act')(x)

    x = Dropout(0.5)(x)

    return x






