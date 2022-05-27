# -*- coding:utf-8 -*-
from model_profiler import model_profiler

import tensorflow as tf
# 보조 loss를 마지막 loss전에 만들고, 이는 배경을 중점적으로 학습하는 곳이고 이 feature 를 최종 output에 attention하여 최종 loss를 만들자
def multi_scale_network(input_shape=(512, 512, 3), nclasses=1):

    h = tf.keras.Input(input_shape)

    model = tf.keras.applications.DenseNet121(include_top=False, input_shape=input_shape)

    h = model.output

    block_3 = model.get_layer('pool4_relu').output  # 32, 32, 1024
    block_3 = tf.keras.layers.Conv2D(filters=1024, kernel_size=1, use_bias=False, groups=2)(block_3)
    block_3 = tf.keras.layers.BatchNormalization()(block_3)
    block_3 = tf.keras.layers.ReLU()(block_3)
    block_3_h_1 = block_3[:, :, :, 0:512]
    block_3_h_2 = block_3[:, :, :, 512:]
    block_2 = model.get_layer('pool3_relu').output  # 64, 64, 512
    block_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=1, use_bias=False, groups=2)(block_2)
    block_2 = tf.keras.layers.BatchNormalization()(block_2)
    block_2 = tf.keras.layers.ReLU()(block_2)
    block_2_h_1 = block_2[:, :, :, 0:256]
    block_2_h_2 = block_2[:, :, :, 256:]
    block_1 = model.get_layer('pool2_relu').output  # 128, 128, 256
    block_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, use_bias=False, groups=2)(block_1)
    block_1 = tf.keras.layers.BatchNormalization()(block_1)
    block_1 = tf.keras.layers.ReLU()(block_1)
    block_1_h_1 = block_1[:, :, :, 0:128]
    block_1_h_2 = block_1[:, :, :, 128:]
    block_0 = model.get_layer('conv1/relu').output  # 256, 256, 64
    block_0 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, use_bias=False, groups=2)(block_0)
    block_0 = tf.keras.layers.BatchNormalization()(block_0)
    block_0 = tf.keras.layers.ReLU()(block_0)
    block_0_h_1 = block_0[:, :, :, 0:32]
    block_0_h_2 = block_0[:, :, :, 32:]

    h = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding="same", use_bias=False, groups=2)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding="same", use_bias=False, groups=2)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h_1 = h[:, :, :, 0:512]
    h_2 = h[:, :, :, 512:]

    h_1 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.concat([h_1, block_3_h_1], -1)
    h_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same")(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding="same", use_bias=False)(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)
    h_2 = tf.concat([h_2, block_3_h_2], -1)
    h_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same")(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    h_1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.concat([h_1, block_2_h_1], -1)
    h_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same")(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding="same", use_bias=False)(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)
    h_2 = tf.concat([h_2, block_2_h_2], -1)
    h_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same")(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    h_1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.concat([h_1, block_1_h_1], -1)
    h_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same")(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="same", use_bias=False)(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)
    h_2 = tf.concat([h_2, block_1_h_2], -1)
    h_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same")(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    h_1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.concat([h_1, block_0_h_1], -1)
    h_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="same", use_bias=False)(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)
    h_2 = tf.concat([h_2, block_0_h_2], -1)
    h_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    h_1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1)(h_1)

    h_2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding="same", use_bias=False)(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)
    h_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)
    h_2 = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1)(h_2)

    #h_1 = tf.image.resize(h_1, [input_shape[0], input_shape[1]])
    #h_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(h_1)
    #h_1 = tf.keras.layers.BatchNormalization()(h_1)
    #h_1 = tf.keras.layers.ReLU()(h_1)
    #h_1 = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1)(h_1)

    #h_2 = tf.image.resize(h_2, [input_shape[0], input_shape[1]])
    #h_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(h_2)
    #h_2 = tf.keras.layers.BatchNormalization()(h_2)
    #h_2 = tf.keras.layers.ReLU()(h_2)
    #h_2 = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1)(h_2)

    # background (h_1)에 대해 dice를 하고, h_1에 sigmoid를 한 후 1을 뺴준뒤 h_2에 곱해주어 최종 h_2에 대한 loss 및 테스틀 진행
    # 이거 꼭 기억해!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # 오늘 집가서 꼭 돌려봐!!
    return tf.keras.Model(inputs=model.input, outputs=[h_1, h_2])

mo = multi_scale_network()
pro = model_profiler(mo, 5)
mo.summary()
print(pro)
