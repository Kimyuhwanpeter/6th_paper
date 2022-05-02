# -*- coding:utf-8 -*-
from tensorflow.keras import backend as K
from tensorflow.keras.layers import AveragePooling2D, Lambda, Conv2D, Conv2DTranspose, Activation, Reshape, concatenate, Concatenate, BatchNormalization, ZeroPadding2D

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

    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]))(tensor)
    y_pool = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu',)(y_pool)

    y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = Conv2D(filters=filters, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(tensor)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = Conv2D(filters=filters, kernel_size=3, dilation_rate=6, padding='same', use_bias=False)(tensor)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = Conv2D(filters=filters, kernel_size=3, dilation_rate=12, padding='same', use_bias=False)(tensor)
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    y_18 = Conv2D(filters=filters, kernel_size=3, dilation_rate=18, padding='same', use_bias=False)(tensor)
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation('relu')(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18])

    y = Conv2D(filters=filters, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = tf.nn.softmax(y, -1)
    #y = Activation(tf.nn.softmax, name='softmax_final'+ name)(y)
    return y


def modified_model(inputs, classes=2):
    
    h = inputs

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False)(h_1)    # downsample
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding="same", use_bias=False)(h_2)    # downsample
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    
    h_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)    

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=2, padding="same", use_bias=False)(h_3)    # downsample
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    #########################################################################################################################
    h_3 = ASPP(h_3, 256)
    h = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h) * h_3

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h_2 = ASPP(h_2, 128)
    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h) * h_2

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h_1 = ASPP(h_1, 64)
    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h) * h_1

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=classes, kernel_size=1)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def pateh_model(input_shape=(896, 896, 3)):

    h_1 = inputs_1 = tf.keras.Input(input_shape, name="input_1")
    h_2 = inputs_2 = tf.keras.Input(input_shape, name="input_2")
    h_3 = inputs_3 = tf.keras.Input(input_shape, name="input_3")
    h_4 = inputs_4 = tf.keras.Input(input_shape, name="input_4")

    model_1 = modified_model(h_1)
    model_2 = modified_model(h_2)
    model_3 = modified_model(h_3)
    model_4 = modified_model(h_4)
    

    return tf.keras.Model(inputs=[inputs_1, inputs_2, inputs_3, inputs_4], outputs=[model_1.output, model_2.output, model_3.output, model_4.output])

mo = pateh_model()
mo.summary()
