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
    temp_h = tf.image.resize(temp_h, [input_shape[0] // 16, input_shape[1] // 16]) * block_4
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
    output_4 = tf.image.resize(h, [input_shape[0], input_shape[1]])
    output_4 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(output_4)
    output_4 = tf.keras.layers.BatchNormalization()(output_4)
    output_4 = tf.keras.layers.ReLU()(output_4)

    temp_h = tf.keras.layers.Conv2D(filters=512, kernel_size=1, use_bias=False)(block_4)
    temp_h = tf.keras.layers.BatchNormalization()(temp_h)
    temp_h = tf.nn.sigmoid(temp_h)
    temp_h = tf.image.resize(temp_h, [input_shape[0] // 8, input_shape[1] // 8]) * block_3
    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, temp_h], -1)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    output_3 = tf.image.resize(h, [input_shape[0], input_shape[1]])
    output_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(output_3)
    output_3 = tf.keras.layers.BatchNormalization()(output_3)
    output_3 = tf.keras.layers.ReLU()(output_3)

    temp_h = tf.keras.layers.Conv2D(filters=256, kernel_size=1, use_bias=False)(block_3)
    temp_h = tf.keras.layers.BatchNormalization()(temp_h)
    temp_h = tf.nn.sigmoid(temp_h)
    temp_h = tf.image.resize(temp_h, [input_shape[0] // 4, input_shape[1] // 4]) * block_2
    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, temp_h], -1)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    output_2 = tf.image.resize(h, [input_shape[0], input_shape[1]])
    output_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(output_2)
    output_2 = tf.keras.layers.BatchNormalization()(output_2)
    output_2 = tf.keras.layers.ReLU()(output_2)

    temp_h = tf.keras.layers.Conv2D(filters=128, kernel_size=1, use_bias=False)(block_2)
    temp_h = tf.keras.layers.BatchNormalization()(temp_h)
    temp_h = tf.nn.sigmoid(temp_h)
    temp_h = tf.image.resize(temp_h, [input_shape[0] // 2, input_shape[1] // 2]) * block_1
    h = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, temp_h], -1)
    h = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    output_1 = tf.image.resize(h, [input_shape[0], input_shape[1]])
    output_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(output_1)
    output_1 = tf.keras.layers.BatchNormalization()(output_1)
    output_1 = tf.keras.layers.ReLU()(output_1)

    h = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, output_1, output_2, output_3, output_4], -1)
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1)(h)

    model = tf.keras.Model(inputs=model.input, outputs=h)

    return model

mo = multi_scale_network(input_shape=(512, 512, 3)) # ???? ?????Ŀ? 896 ?????? ??  ??ġ 4?? ?н??غ???!!! ??????!!!!!!!!!!!!!!
mo.summary()
profiler = model_profiler(mo, 7)
print(profiler)
