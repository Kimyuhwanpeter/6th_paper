# -*- coding:utf-8 -*-
from model_profiler import model_profiler
import tensorflow as tf


def batchnorm_relu(input):

    h = tf.keras.layers.BatchNormalization()(input)
    h = tf.keras.layers.ReLU()(h)

    return h

def modified_Unet_PP(input_shape=(512, 512, 3)):

    h = inputs = tf.keras.Input(input_shape)

    encoder_backbone = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False)


    h_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="block1_conv1")(h)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="block1_conv2")(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_1)

    h_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", name="block2_conv1")(pool_1)
    h_2 = tf.keras.layers.ReLU()(h_2)
    h_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", name="block2_conv2")(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_2)

    h_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="block3_conv1")(pool_2)
    h_3 = tf.keras.layers.ReLU()(h_3)
    h_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="block3_conv2")(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)
    h_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="block3_conv3")(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_3)

    h_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="block4_conv1")(pool_3)
    h_4 = tf.keras.layers.ReLU()(h_4)
    h_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="block4_conv2")(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)
    h_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="block4_conv3")(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)

    pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_4)

    h_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="block5_conv1")(pool_4)
    h_5 = tf.keras.layers.ReLU()(h_5)
    h_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="block5_conv2")(h_5)
    h_5 = tf.keras.layers.ReLU()(h_5)
    h_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="block5_conv3")(h_5)
    h_5 = tf.keras.layers.ReLU()(h_5)

    pool_5 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_5)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(pool_5)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    #######################################################################################################
    # from encoder
    h_1_upsample_1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(pool_1)

    h_2_upsample_1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(pool_2)
    h_2_upsample_2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(batchnorm_relu(h_2_upsample_1))

    h_3_upsample_1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, use_bias=False)(pool_3)
    h_3_upsample_2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(batchnorm_relu(h_3_upsample_1))
    h_3_upsample_3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(batchnorm_relu(h_3_upsample_2))

    h_4_upsample_1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, use_bias=False)(pool_4)
    h_4_upsample_2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, use_bias=False)(batchnorm_relu(h_4_upsample_1))
    h_4_upsample_3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(batchnorm_relu(h_4_upsample_2))
    h_4_upsample_4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(batchnorm_relu(h_4_upsample_3))

    #######################################################################################################
    
    h = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h= tf.concat([h_5, h], -1)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    
    de_h_4_upsample_1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, use_bias=False)(h)
    de_h_4_upsample_2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, use_bias=False)(batchnorm_relu(de_h_4_upsample_1))
    de_h_4_upsample_3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(batchnorm_relu(de_h_4_upsample_2))
    de_h_4_upsample_4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(batchnorm_relu(de_h_4_upsample_3))

    h_4_upsample_1 = tf.where(tf.nn.sigmoid(h_4_upsample_1) >= 0.5, h_4_upsample_1, 0.)
    de_h_4_upsample_1 = tf.where(tf.nn.sigmoid(de_h_4_upsample_1) >= 0.5, de_h_4_upsample_1, 0.)
    sum_h_4 = (h_4_upsample_1) + (de_h_4_upsample_1)
    sum_h_4 = tf.keras.layers.BatchNormalization()(sum_h_4)
    sum_h_4 = tf.keras.layers.ReLU()(sum_h_4)

    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h_4, sum_h_4, h], -1)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    de_h_3_upsample_1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, use_bias=False)(h)
    de_h_3_upsample_2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(batchnorm_relu(de_h_3_upsample_1))
    de_h_3_upsample_3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(batchnorm_relu(de_h_3_upsample_2))

    h_4_upsample_2 = tf.where(tf.nn.sigmoid(h_4_upsample_2) >= 0.5, h_4_upsample_2, 0.)
    de_h_3_upsample_1 = tf.where(tf.nn.sigmoid(de_h_3_upsample_1) >= 0.5, de_h_3_upsample_1, 0.)
    sum_h_3 = (h_4_upsample_2) + (de_h_3_upsample_1)
    sum_h_3 = tf.keras.layers.BatchNormalization()(sum_h_3)
    sum_h_3_2 = tf.keras.layers.ReLU()(sum_h_3)

    h_3_upsample_1 = tf.where(tf.nn.sigmoid(h_3_upsample_1) >= 0.5, h_3_upsample_1, 0.)
    de_h_4_upsample_2 = tf.where(tf.nn.sigmoid(de_h_4_upsample_2) >= 0.5, de_h_4_upsample_2, 0.)
    sum_h_3 = (h_3_upsample_1) + (de_h_4_upsample_2)
    sum_h_3 = tf.keras.layers.BatchNormalization()(sum_h_3)
    sum_h_3_1 = tf.keras.layers.ReLU()(sum_h_3)

    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h_3, sum_h_3_1, sum_h_3_2, h], -1)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    de_h_2_upsample_1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(h)
    de_h_2_upsample_2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(batchnorm_relu(de_h_2_upsample_1))

    h_4_upsample_3 = tf.where(tf.nn.sigmoid(h_4_upsample_3) >= 0.5, h_4_upsample_3, 0.)
    de_h_2_upsample_1 = tf.where(tf.nn.sigmoid(de_h_2_upsample_1) >= 0.5, de_h_2_upsample_1, 0.)
    sum_h_2 = (h_4_upsample_3) + (de_h_2_upsample_1)
    sum_h_2 = tf.keras.layers.BatchNormalization()(sum_h_2)
    sum_h_2_3 = tf.keras.layers.ReLU()(sum_h_2)

    h_3_upsample_2 = tf.where(tf.nn.sigmoid(h_3_upsample_2) >= 0.5, h_3_upsample_2, 0.)
    de_h_3_upsample_2 = tf.where(tf.nn.sigmoid(de_h_3_upsample_2) >= 0.5, de_h_3_upsample_2, 0.)
    sum_h_2 = (h_3_upsample_2) + (de_h_3_upsample_2)
    sum_h_2 = tf.keras.layers.BatchNormalization()(sum_h_2)
    sum_h_2_2 = tf.keras.layers.ReLU()(sum_h_2)

    h_2_upsample_1 = tf.where(tf.nn.sigmoid(h_2_upsample_1) >= 0.5, h_2_upsample_1, 0.)
    de_h_4_upsample_3 = tf.where(tf.nn.sigmoid(de_h_4_upsample_3) >= 0.5, de_h_4_upsample_3, 0.)
    sum_h_2 = (h_2_upsample_1) + (de_h_4_upsample_3)
    sum_h_2 = tf.keras.layers.BatchNormalization()(sum_h_2)
    sum_h_2_1 = tf.keras.layers.ReLU()(sum_h_2)

    h = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h_2, sum_h_2_1, sum_h_2_2, sum_h_2_3, h], -1)
    h = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    de_h_1_upsample_1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(h)

    h_4_upsample_4 = tf.where(tf.nn.sigmoid(h_4_upsample_4) >= 0.5, h_4_upsample_4, 0.)
    de_h_1_upsample_1 = tf.where(tf.nn.sigmoid(de_h_1_upsample_1) >= 0.5, de_h_1_upsample_1, 0.)
    sum_h_1 = (h_4_upsample_4) + (de_h_1_upsample_1)
    sum_h_1 = tf.keras.layers.BatchNormalization()(sum_h_1)
    sum_h_1_4 = tf.keras.layers.ReLU()(sum_h_1)

    h_3_upsample_3 = tf.where(tf.nn.sigmoid(h_3_upsample_3) >= 0.5, h_3_upsample_3, 0.)
    de_h_3_upsample_3 = tf.where(tf.nn.sigmoid(de_h_3_upsample_3) >= 0.5, de_h_3_upsample_3, 0.)
    sum_h_1 = (h_3_upsample_3) + (de_h_3_upsample_3)
    sum_h_1 = tf.keras.layers.BatchNormalization()(sum_h_1)
    sum_h_1_3 = tf.keras.layers.ReLU()(sum_h_1)

    h_2_upsample_2 = tf.where(tf.nn.sigmoid(h_2_upsample_2) >= 0.5, h_2_upsample_2, 0.)
    de_h_2_upsample_2 = tf.where(tf.nn.sigmoid(de_h_2_upsample_2) >= 0.5, de_h_2_upsample_2, 0.)
    sum_h_1 = (h_2_upsample_2) + (de_h_2_upsample_2)
    sum_h_1 = tf.keras.layers.BatchNormalization()(sum_h_1)
    sum_h_1_2 = tf.keras.layers.ReLU()(sum_h_1)

    h_1_upsample_1 = tf.where(tf.nn.sigmoid(h_1_upsample_1) >= 0.5, h_1_upsample_1, 0.)
    de_h_4_upsample_4 = tf.where(tf.nn.sigmoid(de_h_4_upsample_4) >= 0.5, de_h_4_upsample_4, 0.)
    sum_h_1 = (h_1_upsample_1) + (de_h_4_upsample_4)
    sum_h_1 = tf.keras.layers.BatchNormalization()(sum_h_1)
    sum_h_1_1 = tf.keras.layers.ReLU()(sum_h_1)

    h = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h_1, sum_h_1_1, sum_h_1_2, sum_h_1_3, sum_h_1_4, h], -1)
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    sum_h_1_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(sum_h_1_1)
    sum_h_1_1 = tf.keras.layers.BatchNormalization()(sum_h_1_1)
    sum_h_1_1 = tf.keras.layers.ReLU()(sum_h_1_1)

    sum_h_1_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(sum_h_1_2)
    sum_h_1_2 = tf.keras.layers.BatchNormalization()(sum_h_1_2)
    sum_h_1_2 = tf.keras.layers.ReLU()(sum_h_1_2)

    sum_h_1_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(sum_h_1_3)
    sum_h_1_3 = tf.keras.layers.BatchNormalization()(sum_h_1_3)
    sum_h_1_3 = tf.keras.layers.ReLU()(sum_h_1_3)

    sum_h_1_4 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(sum_h_1_4)
    sum_h_1_4 = tf.keras.layers.BatchNormalization()(sum_h_1_4)
    sum_h_1_4 = tf.keras.layers.ReLU()(sum_h_1_4)

    h = (h + sum_h_1_1 + sum_h_1_2 + sum_h_1_3 + sum_h_1_4) / 5.

    h = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(h)

    model = tf.keras.Model(inputs=inputs, outputs=h)

    return model

mo = modified_Unet_PP()
pro = model_profiler(mo, 2)
mo.summary()
print(pro)
