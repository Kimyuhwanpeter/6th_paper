# -*- coding:utf-8 -*-
from tensorflow.keras import backend as K
from model_profiler import model_profiler
import tensorflow as tf
import other_utils.resnext as resnext
# Backbone을 VGG-16으로 설정하고, 뒷 단을 배경 및 object에 대한 decoder를 만들자

def Upsample(tensor, size):
    '''bilinear upsampling'''

    def bilinear_upsample(x, size):
        resized = tf.image.resize(
            images=x, size=size)
        return resized
    y = tf.keras.layers.Lambda(lambda x: bilinear_upsample(x, size),output_shape=size)(tensor)

    return y

def ASPP(tensor):
    '''atrous spatial pyramid pooling'''
    dims = K.int_shape(tensor)

    y_pool = tf.keras.layers.AveragePooling2D(pool_size=(dims[1], dims[2]))(tensor)
    y_pool = tf.keras.layers.Conv2D(filters=512, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = tf.keras.layers.BatchNormalization()(y_pool)
    y_pool = tf.keras.layers.Activation('relu')(y_pool)

    y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(tensor)
    y_1 = tf.keras.layers.BatchNormalization()(y_1)
    y_1 = tf.keras.layers.Activation('relu')(y_1)

    y_6 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, dilation_rate=12, padding='same', use_bias=False)(tensor)
    y_6 = tf.keras.layers.BatchNormalization()(y_6)
    y_6 = tf.keras.layers.Activation('relu')(y_6)

    y_12 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, dilation_rate=24, padding='same', use_bias=False)(tensor)
    y_12 = tf.keras.layers.BatchNormalization()(y_12)
    y_12 = tf.keras.layers.Activation('relu')(y_12)

    y_18 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, dilation_rate=36, padding='same', use_bias=False)(tensor)
    y_18 = tf.keras.layers.BatchNormalization()(y_18)
    y_18 = tf.keras.layers.Activation('relu')(y_18)

    y = tf.concat([y_pool, y_1, y_6, y_12, y_18], -1)

    y = tf.keras.layers.Conv2D(filters=512, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)

    return y

def modified_seg_model(input_shape=(512, 512, 3), nclasses=2):

    h = inputs = tf.keras.Input(input_shape)

    model = resnext.ResNeXt50(input_shape=input_shape, include_top=False)

    low_level_feature = model.get_layer('stage2_unit1_relu1').output
    #low_level_feature = tf.keras.layers.Conv2D(filters=)
    non_ob_low_level_features = low_level_feature[:, :, :, 0:128]
    non_ob_low_level_features = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(non_ob_low_level_features)
    non_ob_low_level_features = tf.keras.layers.BatchNormalization()(non_ob_low_level_features)
    non_ob_low_level_features = tf.keras.layers.ReLU()(non_ob_low_level_features)
    ob_low_level_features = low_level_feature[:, :, :, 128:]
    ob_low_level_features = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(ob_low_level_features)
    ob_low_level_features = tf.keras.layers.BatchNormalization()(ob_low_level_features)
    ob_low_level_features = tf.keras.layers.ReLU()(ob_low_level_features)

    h = model.get_layer('stage4_unit1_relu1').output

    # devide decoder part
    non_ob_h = h[:, :, :, 0:512]
    non_ob_h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(non_ob_h)
    non_ob_h = tf.keras.layers.BatchNormalization()(non_ob_h)
    non_ob_h = tf.keras.layers.ReLU()(non_ob_h)
    ob_h = h[:, :, :, 512:]
    ob_h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(ob_h)
    ob_h = tf.keras.layers.BatchNormalization()(ob_h)
    ob_h = tf.keras.layers.ReLU()(ob_h)

    # decoder
    non_ob_h = ASPP(non_ob_h)
    non_ob_h = Upsample(non_ob_h, size=[input_shape[0] // 4, input_shape[1] // 4])
    ob_h = ASPP(ob_h)
    ob_h = Upsample(ob_h, size=[input_shape[0] // 4, input_shape[1] // 4])

    non_ob_low_level_features = tf.keras.layers.Conv2D(filters=48, kernel_size=1, use_bias=False)(non_ob_low_level_features)
    non_ob_low_level_features = tf.keras.layers.BatchNormalization()(non_ob_low_level_features)
    non_ob_low_level_features = tf.keras.layers.ReLU()(non_ob_low_level_features)

    ob_low_level_features = tf.keras.layers.Conv2D(filters=48, kernel_size=1, use_bias=False)(ob_low_level_features)
    ob_low_level_features = tf.keras.layers.BatchNormalization()(ob_low_level_features)
    ob_low_level_features = tf.keras.layers.ReLU()(ob_low_level_features)

    b_h = tf.concat([non_ob_h, non_ob_low_level_features], -1)
    b_h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(b_h)
    b_h = tf.keras.layers.BatchNormalization()(b_h)
    b_h = tf.keras.layers.ReLU()(b_h)
    b_h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(b_h)
    b_h = tf.keras.layers.BatchNormalization()(b_h)
    b_h = tf.keras.layers.ReLU()(b_h)
    b_h = Upsample(b_h, size=[input_shape[0], input_shape[1]])
    b_h = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(b_h)

    o_h = tf.concat([ob_h, ob_low_level_features], -1)
    o_h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(o_h)
    o_h = tf.keras.layers.BatchNormalization()(o_h)
    o_h = tf.keras.layers.ReLU()(o_h)
    o_h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(o_h)
    o_h = tf.keras.layers.BatchNormalization()(o_h)
    o_h = tf.keras.layers.ReLU()(o_h)
    o_h = Upsample(o_h, size=[input_shape[0], input_shape[1]])
    o_h = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(o_h)

    h = tf.concat([b_h, o_h], -1)

    # decoder 1
    
    model = tf.keras.Model(inputs=model.input, outputs=h)
    #backbone = tf.keras.applications.VGG16(input_shape=(224, 224, 3))

    #model.get_layer("conv1").set_weights(backbone.get_layer("block1_conv1").get_weights())
    #model.get_layer("conv2").set_weights(backbone.get_layer("block1_conv2").get_weights())
    #model.get_layer("conv3").set_weights(backbone.get_layer("block2_conv1").get_weights())
    #model.get_layer("conv4").set_weights(backbone.get_layer("block2_conv2").get_weights())
    #model.get_layer("conv5").set_weights(backbone.get_layer("block2_conv2").get_weights())
    #model.get_layer("conv6").set_weights(backbone.get_layer("block3_conv1").get_weights())
    #model.get_layer("conv7").set_weights(backbone.get_layer("block3_conv2").get_weights())
    #model.get_layer("conv8").set_weights(backbone.get_layer("block3_conv3").get_weights())
    ##model.get_layer("conv9").set_weights(backbone.get_layer("block3_conv3").get_weights())
    #model.get_layer("conv10").set_weights(backbone.get_layer("block4_conv1").get_weights())
    #model.get_layer("conv11").set_weights(backbone.get_layer("block4_conv2").get_weights())
    #model.get_layer("conv12").set_weights(backbone.get_layer("block4_conv3").get_weights())
    #model.get_layer("conv13").set_weights(backbone.get_layer("block4_conv3").get_weights())

    return model

mo = modified_seg_model(input_shape=(512, 512, 3))
model_pro = model_profiler(mo, 4)
mo.summary()
print(model_pro)
