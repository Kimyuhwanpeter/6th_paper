# -*- coding:utf-8 -*-
from tensorflow.keras import backend as K
from tensorflow.keras.layers import AveragePooling2D, Lambda, Conv2D, Conv2DTranspose, Activation, Reshape, concatenate, Concatenate, BatchNormalization, ZeroPadding2D, DepthwiseConv2D

import tensorflow as tf


def Upsample(tensor, size):
    '''bilinear upsampling'''

    def bilinear_upsample(x, size):
        resized = tf.image.resize(
            images=x, size=size)
        return resized
    y = Lambda(lambda x: bilinear_upsample(x, size),
               output_shape=size)(tensor)
    return y

def ASPP(tensor, filters):
    '''atrous spatial pyramid pooling'''
    dims = K.int_shape(tensor)

    split_dim = dims[3] // 4

    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]))(tensor[:, :, :, 0*split_dim:1*split_dim])
    y_pool = Conv2D(filters=split_dim, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu',)(y_pool)

    y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = Conv2D(filters=split_dim, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(tensor[:, :, :, 1*split_dim:2*split_dim])
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = Conv2D(filters=split_dim, kernel_size=3, dilation_rate=6, padding='same', use_bias=False)(tensor[:, :, :, 2*split_dim:3*split_dim])
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = Conv2D(filters=split_dim, kernel_size=3, dilation_rate=12, padding='same', use_bias=False)(tensor[:, :, :, 3*split_dim:4*split_dim])
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    # y_18 = Conv2D(filters=filters, kernel_size=3, dilation_rate=18, padding='same', use_bias=False)(tensor)
    # y_18 = BatchNormalization()(y_18)
    # y_18 = Activation('relu')(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12])

    y = Conv2D(filters=filters, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    #y = tf.nn.softmax(y, -1)
    y = Activation("relu")(y)
    return y

def attention_ASPP(tensor, filters):
    '''atrous spatial pyramid pooling'''
    dims = K.int_shape(tensor)

    split_dim = dims[3] // 4

    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]))(tensor[:, :, :, 0*split_dim:1*split_dim])
    y_pool = DepthwiseConv2D(kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu',)(y_pool)

    y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = Conv2D(filters=split_dim, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(tensor[:, :, :, 1*split_dim:2*split_dim])
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = Conv2D(filters=split_dim, kernel_size=3, dilation_rate=6, padding='same', use_bias=False)(tensor[:, :, :, 2*split_dim:3*split_dim])
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = Conv2D(filters=split_dim, kernel_size=3, dilation_rate=12, padding='same', use_bias=False)(tensor[:, :, :, 3*split_dim:4*split_dim])
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    # y_18 = DepthwiseConv2D(kernel_size=3, dilation_rate=18, padding='same', use_bias=False)(tensor)
    # y_18 = BatchNormalization()(y_18)
    # y_18 = Activation('relu')(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12])

    y = Conv2D(filters=filters, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    # y = DepthwiseConv2D(kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = tf.reduce_mean(y, -1, keepdims=True)
    y = tf.nn.sigmoid(y)
    return y

def modified_model(input_shape=(512, 512, 3), classes=2):
    
    h = inputs = tf.keras.Input(input_shape)

    h_att_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False, name="conv1")(h)
    h_att_1 = tf.keras.layers.BatchNormalization()(h_att_1)
    h_att_1 = tf.keras.layers.ReLU()(h_att_1)

    h_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False, name="conv2")(h_att_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    # h = tf.pad(h_1, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    # h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, use_bias=False)(h)    # downsample
    # h = tf.keras.layers.BatchNormalization()(h)
    # h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_1) # downsample

    h_att_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False, name="conv3")(h)
    h_att_2 = tf.keras.layers.BatchNormalization()(h_att_2)
    h_att_2 = tf.keras.layers.ReLU()(h_att_2)

    h_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False, name="conv4")(h_att_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    # h = tf.pad(h_2, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    # h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, use_bias=False)(h)    # downsample
    # h = tf.keras.layers.BatchNormalization()(h)
    # h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_2) # downsample

    h_att_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False, name="conv5")(h)
    h_att_3 = tf.keras.layers.BatchNormalization()(h_att_3)
    h_att_3 = tf.keras.layers.ReLU()(h_att_3)
    
    h_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False, name="conv6")(h_att_3)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    # h = tf.pad(h_3, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    # h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=2, use_bias=False)(h)    # downsample
    # h = tf.keras.layers.BatchNormalization()(h)
    # h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_3) # downsample

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False, name="conv7")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False, name="conv8")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    encoder_2 = h

    #########################################################################################################################
    h_3 = ASPP(h_3, 256)
    h = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h_3, h], -1)
    h_att_3 = attention_ASPP(h_att_3, 256)
    h = h_att_3 * h

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h_2 = ASPP(h_2, 128)
    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h_2, h], -1)
    h_att_2 = attention_ASPP(h_att_2, 128)
    h = h_att_2 * h

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h_1 = ASPP(h_1, 64)
    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h_1, h], -1)
    h_att_1 = attention_ASPP(h_att_1, 64)
    h = h_att_1 * h

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=classes, kernel_size=1)(h) # object
    decoder_output_1 = h
    #########################################################################################################################

    #encoder_2_1 = ASPP(encoder_2[:, :, :, 0:256], 256)
    #encoder_2_1 = Upsample(encoder_2_1, [K.int_shape(encoder_2)[1] * 2, K.int_shape(encoder_2)[2] * 2])
    
    #encoder_2_2 = ASPP(encoder_2[:, :, :, 256:], 256)
    #encoder_2_2 = Upsample(encoder_2_2, [K.int_shape(encoder_2)[1] * 2, K.int_shape(encoder_2)[2] * 2])

    #decoder_2 = tf.concat([encoder_2_1, encoder_2_2], -1)
    #decoder_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(decoder_2)
    #decoder_2 = tf.keras.layers.BatchNormalization()(decoder_2)
    #decoder_2 = tf.keras.layers.ReLU()(decoder_2)
    #decoder_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(decoder_2)
    #decoder_2 = tf.keras.layers.BatchNormalization()(decoder_2)
    #decoder_2 = tf.keras.layers.ReLU()(decoder_2)  # ???????? ?????????

    return tf.keras.Model(inputs=inputs, outputs=h)
