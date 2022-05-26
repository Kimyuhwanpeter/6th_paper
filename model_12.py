# -*- coding:utf-8 -*-
import tensorflow as tf
from model_profiler import model_profiler

def multi_scale_network(input_shape=(512, 512, 3), nclasses=2):

    model = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False)

    h = model.output

    block_4 = model.get_layer('block5_conv3').output    # [32, 32, 512]
    block_3 = model.get_layer('block4_conv3').output    # [64, 64, 512]
    block_2 = model.get_layer('block3_conv2').output    # [128, 128, 256]
    block_1 = model.get_layer('block2_conv2').output    # [256, 256, 128]

    temp_h = tf.keras.layers.Conv2D(filters=512, kernel_size=1, use_bias=False)(h)
    temp_h = tf.keras.layers.BatchNormalization()(temp_h)
    temp_h = tf.nn.sigmoid(temp_h)
    temp_h = tf.image.resize(temp_h, [32, 32]) * block_4
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [16, 16, 512]
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [16, 16, 512]
    
    h = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, temp_h], -1)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    temp_h = tf.keras.layers.Conv2D(filters=512, kernel_size=1, use_bias=False)(block_4)
    temp_h = tf.keras.layers.BatchNormalization()(temp_h)
    temp_h = tf.nn.sigmoid(temp_h)
    temp_h = tf.image.resize(temp_h, [64, 64]) * block_3
    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, temp_h], -1)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    temp_h = tf.keras.layers.Conv2D(filters=256, kernel_size=1, use_bias=False)(block_3)
    temp_h = tf.keras.layers.BatchNormalization()(temp_h)
    temp_h = tf.nn.sigmoid(temp_h)
    temp_h = tf.image.resize(temp_h, [128, 128]) * block_2
    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, temp_h], -1)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    temp_h = tf.keras.layers.Conv2D(filters=128, kernel_size=1, use_bias=False)(block_2)
    temp_h = tf.keras.layers.BatchNormalization()(temp_h)
    temp_h = tf.nn.sigmoid(temp_h)
    temp_h = tf.image.resize(temp_h, [256, 256]) * block_1
    h = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, temp_h], -1)
    h = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1)(h)

    model = tf.keras.Model(inputs=model.input, outputs=h)

    return model

mo = multi_scale_network()
mo.summary()
profiler = model_profiler(mo, 16)
print(profiler)
