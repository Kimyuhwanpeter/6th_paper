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

def ASPP(tensor, filters, name):
    '''atrous spatial pyramid pooling'''
    dims = K.int_shape(tensor)

    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling' + name)(tensor)
    y_pool = Conv2D(filters=filters, kernel_size=1, padding='same', name='pool_1x1conv2d'+ name, use_bias=False)(y_pool)
    y_pool = BatchNormalization(name='bn_1'+ name)(y_pool)
    y_pool = Activation('relu', name='relu_1'+ name)(y_pool)

    y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = Conv2D(filters=filters, kernel_size=1, dilation_rate=1, padding='same', name='ASPP_conv2d_d1'+ name, use_bias=False)(tensor)
    y_1 = BatchNormalization(name='bn_2'+ name)(y_1)
    y_1 = Activation('relu', name='relu_2'+ name)(y_1)

    y_6 = Conv2D(filters=filters, kernel_size=3, dilation_rate=6, padding='same',name='ASPP_conv2d_d6'+ name, use_bias=False)(tensor)
    y_6 = BatchNormalization(name='bn_3'+ name)(y_6)
    y_6 = Activation('relu', name='relu_3'+ name)(y_6)

    y_12 = Conv2D(filters=filters, kernel_size=3, dilation_rate=12, padding='same', name='ASPP_conv2d_d12'+ name, use_bias=False)(tensor)
    y_12 = BatchNormalization(name='bn_4'+ name)(y_12)
    y_12 = Activation('relu', name='relu_4'+ name)(y_12)

    y_18 = Conv2D(filters=filters, kernel_size=3, dilation_rate=18, padding='same', name='ASPP_conv2d_d18'+ name, use_bias=False)(tensor)
    y_18 = BatchNormalization(name='bn_5'+ name)(y_18)
    y_18 = Activation('relu', name='relu_5'+ name)(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat'+ name)

    y = Conv2D(filters=filters, kernel_size=1, dilation_rate=1, padding='same',name='ASPP_conv2d_final'+ name, use_bias=False)(y)
    y = BatchNormalization(name='bn_final'+ name)(y)
    y = Activation('relu', name='relu_final'+ name)(y)
    return y


def patch_model(input_shape=(512, 512, 3), classes=2):

    h_1 = inputs_1 = tf.keras.Input(input_shape, name="input_1")
    h_2 = inputs_2 = tf.keras.Input(input_shape, name="input_2")
    h_3 = inputs_3 = tf.keras.Input(input_shape, name="input_3")
    h_4 = inputs_4 = tf.keras.Input(input_shape, name="input_4")

    #h_1 = h[:, 0:512, 0:512, :]   # h_left_top
    #h_2 = h[:, 0:512, 512:, :]   # h_right_top
    #h_3 = h[:, 512:, 0:512, :]   # h_left_down
    #h_4 = h[:, 512:, 512:, :]   # h_right_down

    ################################################################################################
    model_1 = tf.keras.applications.MobileNetV2(input_shape=input_shape, input_tensor=h_1, include_top=False)
    h_1 = model_1.get_layer("block_16_project_BN").output

    h_1_a = ASPP(h_1, 256, "_1")
    h_1 = tf.concat([h_1, h_1_a], -1)
    h_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = Upsample(tensor=h_1, size=[input_shape[0] // 4, input_shape[1] // 4])
    h_1_b = ASPP(h_1, 128, "_2")

    h_1 = tf.concat([h_1_b, model_1.get_layer("block_3_expand").output], -1)
    h_1 = Upsample(tensor=h_1, size=[input_shape[0] // 4, input_shape[1] // 4])
    h_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = Upsample(tensor=h_1, size=[input_shape[0], input_shape[1]])
    h_1_c = ASPP(h_1, 64, "_3")
    h_1 = tf.keras.layers.Conv2D(filters=classes, kernel_size=1)(h_1_c)
    ################################################################################################

    ################################################################################################
    model_2 = tf.keras.applications.MobileNetV2(input_shape=input_shape, input_tensor=h_2, include_top=False)
    for layer in model_2.layers:
        layer._name = layer.name + str("_2")
    h_2 = model_2.get_layer("block_16_project_BN_2").output

    h_2_a = ASPP(h_2, 256, "_4")
    h_2 = tf.concat([h_2, h_2_a], -1)
    h_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)
    h_2 = Upsample(tensor=h_2, size=[input_shape[0] // 4, input_shape[1] // 4])
    h_2_b = ASPP(h_2, 128, "_5")

    h_2 = tf.concat([h_2_b, model_2.get_layer("block_3_expand_2").output], -1)
    h_2 = Upsample(tensor=h_2, size=[input_shape[0] // 4, input_shape[1] // 4])
    h_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)
    h_2 = Upsample(tensor=h_2, size=[input_shape[0], input_shape[1]])
    h_2_c = ASPP(h_2, 64, "_6")
    h_2 = tf.keras.layers.Conv2D(filters=classes, kernel_size=1)(h_2_c)
    ################################################################################################

    ################################################################################################
    model_3 = tf.keras.applications.MobileNetV2(input_shape=input_shape, input_tensor=h_3, include_top=False)
    for layer in model_3.layers:
        layer._name = layer.name + str("_3")
    h_3 = model_3.get_layer("block_16_project_BN_3").output

    h_3_a = ASPP(h_3, 256, "_7")
    h_3 = tf.concat([h_3, h_3_a], -1)
    h_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_3)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)
    h_3 = Upsample(tensor=h_3, size=[input_shape[0] // 4, input_shape[1] // 4])
    h_3_b = ASPP(h_3, 128, "_8")

    h_3 = tf.concat([h_3_b, model_3.get_layer("block_3_expand_3").output], -1)
    h_3 = Upsample(tensor=h_3, size=[input_shape[0] // 4, input_shape[1] // 4])
    h_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h_3)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)
    h_3 = Upsample(tensor=h_3, size=[input_shape[0], input_shape[1]])
    h_3_c = ASPP(h_3, 64, "_9")
    h_3 = tf.keras.layers.Conv2D(filters=classes, kernel_size=1)(h_3_c)
    ################################################################################################

    ################################################################################################
    model_4 = tf.keras.applications.MobileNetV2(input_shape=input_shape, input_tensor=h_4, include_top=False)
    for layer in model_4.layers:
        layer._name = layer.name + str("_4")
    h_4 = model_4.get_layer("block_16_project_BN_4").output

    h_4_a = ASPP(h_4, 256, "_10")
    h_4 = tf.concat([h_4, h_4_a], -1)
    h_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_4)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)
    h_4 = Upsample(tensor=h_4, size=[input_shape[0] // 4, input_shape[1] // 4])
    h_4_b = ASPP(h_4, 128, "_11")

    h_4 = tf.concat([h_4_b, model_4.get_layer("block_3_expand_4").output], -1)
    h_4 = Upsample(tensor=h_4, size=[input_shape[0] // 4, input_shape[1] // 4])
    h_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h_4)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)
    h_4 = Upsample(tensor=h_4, size=[input_shape[0], input_shape[1]])
    h_4_c = ASPP(h_4, 64, "_12")
    h_4 = tf.keras.layers.Conv2D(filters=classes, kernel_size=1)(h_4_c)
    ################################################################################################


    h_1_2 = tf.concat([h_1, h_2], 2)
    h_3_4 = tf.concat([h_3, h_4], 2)
    h = tf.concat([h_1_2, h_3_4], 1)
    

    return tf.keras.Model(inputs=[inputs_1, inputs_2, inputs_3, inputs_4], outputs=[h_1, h_2, h_3, h_4, h])


#img = tf.io.read_file("D:/[1]DB/[5]4th_paper_DB/Fruit/apple_pear/FlowerLabels_temp/IMG_0248.JPG")
#img = tf.image.decode_png(img, 1)
#img = tf.image.resize(img, [1024, 1024], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#import matplotlib.pyplot as plt
#h_left_top = img[0:512, 0:512, 0]
#h_right_top = img[0:512, 512:, 0]
#h_left_down = img[512:, 0:512, 0]
#h_right_down = img[512:, 512:, 0]
#plt.imshow(h_left_top / 255)
#plt.show()
#plt.imshow(h_right_top / 255)
#plt.show()
#plt.imshow(h_left_down / 255)
#plt.show()
#plt.imshow(h_right_down / 255)
#plt.show()
