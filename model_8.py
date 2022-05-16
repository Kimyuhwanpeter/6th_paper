# -*- coding:utf-8 -*-
from tensorflow.keras.layers import AveragePooling2D, Lambda, Conv2D, Conv2DTranspose, Activation, Reshape, concatenate, Concatenate, BatchNormalization, ZeroPadding2D
from tensorflow.keras import backend as K
import tensorflow as tf

def block(x, filters, kernel_size=3, strides=1, conv_shortcut=True, name=None):

    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(4 * filters, 1, strides=strides, name=name + '_0_conv')(x)
        shortcut = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = tf.keras.layers.Conv2D(filters, 1, strides=strides, name=name + '_1_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = tf.keras.layers.ReLU(name=name + '_1_relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", name=name + '_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = tf.keras.layers.ReLU(name=name + '_2_relu')(x)

    x = tf.keras.layers.Conv2D(4*filters, 1, name=name + '_3_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
    x_wo_relu = x
    x = tf.keras.layers.ReLU(name=name + "_out")(x)

    return x, x_wo_relu 

def stack(x, filters, blocks, strides=2, name=None):

    x, _ = block(x, filters, strides=strides, name=name + '_block1')
    for i in range(2, blocks + 1):
        x, x_wo_relu = block(x, filters, conv_shortcut=False, name=name + '_block' + str(i))

    return x, x_wo_relu

def fine_tune_layers(model, backbone):

    for layer in model.layers:     
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            if layer.name != 'except_2':
                model.get_layer(layer.name).set_weights(backbone.get_layer(layer.name).get_weights())
        elif isinstance(layer, tf.keras.layers.Conv2D):
            if layer.name != 'except_1':
                model.get_layer(layer.name).set_weights(backbone.get_layer(layer.name).get_weights())

    #model.get_layer("conv1_conv").set_weights(backbone.get_layer("conv1_conv").get_weights())
    #model.get_layer("conv2_block1_1_conv").set_weights(backbone.get_layer("conv2_block1_1_conv").get_weights())

    return model

def attention_module(original, upsample, d_rate):

    temp = tf.image.resize(original, [original.shape[1] * 2, original.shape[2] * 2])
    spa_stt = tf.reduce_mean(tf.nn.sigmoid(temp), -1, keepdims=True)

    spa_att_map = upsample * spa_stt

    ch_att_1 = tf.keras.layers.GlobalAveragePooling2D()(tf.keras.layers.Conv2D(filters=original.shape[3] // d_rate,
                                                                               kernel_size=1)(original))
    ch_att_1 = tf.nn.softmax(ch_att_1, -1)
    ch_att_1 = tf.expand_dims(ch_att_1, 1)
    ch_att_1 = tf.expand_dims(ch_att_1, 1)
    ch_att_1 = upsample * ch_att_1

    final = spa_att_map + ch_att_1

    return final

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

    y_pool = AveragePooling2D(pool_size=(dims[1], dims[2]))(tensor)
    y_pool = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization(epsilon=1.001e-5)(y_pool)
    y_pool = Activation('relu')(y_pool)

    y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = Conv2D(filters=filters, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(tensor)
    y_1 = BatchNormalization(epsilon=1.001e-5)(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = Conv2D(filters=filters, kernel_size=3, dilation_rate=6, padding='same',use_bias=False)(tensor)
    y_6 = BatchNormalization(epsilon=1.001e-5)(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = Conv2D(filters=filters, kernel_size=3, dilation_rate=12, padding='same', use_bias=False)(tensor)
    y_12 = BatchNormalization(epsilon=1.001e-5)(y_12)
    y_12 = Activation('relu')(y_12)

    y_18 = Conv2D(filters=filters, kernel_size=3, dilation_rate=18, padding='same')(tensor)
    y_18 = BatchNormalization(epsilon=1.001e-5)(y_18)
    y_18 = Activation('relu')(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18])

    y = Conv2D(filters=filters, kernel_size=1, dilation_rate=1, padding='same',use_bias=False)(y)
    y = BatchNormalization(epsilon=1.001e-5)(y)
    y = Activation('relu')(y)
    return y

def multi_scale_network_1(input_shape=(512, 512, 3), nclasses=1):

    backbone = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False)
    x = backbone.get_layer("conv4_block6_out").output
    backbone = tf.keras.Model(backbone.input, x)
    #backbone.summary()

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name='conv1_pad')(h)
    h = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv')(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='conv1_bn')(h)
    h = tf.keras.layers.Activation('relu', name='conv1_relu')(h)
    h = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(h)
    #h = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(h)
    h = tf.keras.layers.Conv2D(64, 3, strides=2, use_bias=True, name='except_1')(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='except_2')(h)
    h = tf.keras.layers.Activation('relu')(h)

    h, h_wo_relu_2 = stack(h, 64, 3, strides=1, name='conv2')
    block_out_2 = h_wo_relu_2
    h, h_wo_relu_1 = stack(h, 128, 4, name='conv3')
    downsample_2 = h_wo_relu_1
    block_out_1 = h_wo_relu_1
    h, h_wo_relu = stack(h, 256, 6, name='conv4')
    downsample_1 = h_wo_relu
    #h = stack(h, 512, 3, name='conv5')

    model = tf.keras.Model(inputs=inputs, outputs=[block_out_2, downsample_2, block_out_1, downsample_1, h])

    model = fine_tune_layers(model, backbone)

    block_out_2, downsample_2, block_out_1, downsample_1, encoder_output = model.output

    h = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=2, strides=2)(encoder_output)
    h = attention_module(downsample_1, h, 2)
    output_1 = h
    output_1 = tf.image.resize(output_1, [input_shape[0], input_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output_1 = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='output_1',
    )(output_1)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_out_1, h], -1)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2)(h)
    h = attention_module(downsample_2, h, 4)
    output_2 = h
    output_2 = tf.image.resize(output_2, [input_shape[0], input_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output_2 = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='output_2',
    )(output_2 * tf.nn.sigmoid(output_1))
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_out_2, h], -1)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    output = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_output',
    )(h * tf.nn.sigmoid(output_2))

    model = tf.keras.Model(inputs=model.input, outputs=[output_1, output_2, output])

    return model

def multi_scale_network_2(input_shape=(512, 512, 3), nclasses=2):

    backbone = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False)
    x = backbone.get_layer("conv4_block6_out").output
    backbone = tf.keras.Model(backbone.input, x)
    #backbone.summary()

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name='conv1_pad')(h)
    h = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv')(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='conv1_bn')(h)
    h = tf.keras.layers.Activation('relu', name='conv1_relu')(h)
    h = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(h)
    #h = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(h)
    h = tf.keras.layers.Conv2D(64, 3, strides=2, use_bias=True, name='except_1')(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='except_2')(h)
    h = tf.keras.layers.Activation('relu')(h)

    h, h_wo_relu_2 = stack(h, 64, 3, strides=1, name='conv2')
    block_out_2 = h_wo_relu_2
    h, h_wo_relu_1 = stack(h, 128, 4, name='conv3')
    downsample_2 = h_wo_relu_1
    block_out_1 = h_wo_relu_1
    h, h_wo_relu = stack(h, 256, 6, name='conv4')
    downsample_1 = h_wo_relu
    #h = stack(h, 512, 3, name='conv5')

    model = tf.keras.Model(inputs=inputs, outputs=[block_out_2, downsample_2, block_out_1, downsample_1, h])

    model = fine_tune_layers(model, backbone)

    block_out_2, downsample_2, block_out_1, downsample_1, encoder_output = model.output

    h = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=2, strides=2)(encoder_output)
    h = attention_module(downsample_1, h, 2)
    output_1 = h
    output_1 = tf.image.resize(output_1, [input_shape[0], input_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output_1 = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='output_1',
    )(output_1)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_out_1, h], -1)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2)(h)
    h = attention_module(downsample_2, h, 4)
    output_2 = h
    output_2 = tf.image.resize(output_2, [input_shape[0], input_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output_2 = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='output_2',
    )(output_2)
    output_2 = output_2 * tf.nn.softmax(output_1, -1)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_out_2, h], -1)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    output = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_output',
    )(h)
    output = output * tf.nn.softmax(output_2, -1)

    model = tf.keras.Model(inputs=model.input, outputs=[output_1, output_2, output])

    return model
