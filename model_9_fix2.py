# -*- coding:utf-8 -*-
import tensorflow as tf

def residual_conv_block1(input, filters, skip=True):

    if skip:
        input = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, kernel_initializer="he_uniform", padding="same", use_bias=False)(input)
        input = tf.keras.layers.BatchNormalization()(input)
        input = tf.keras.layers.ReLU()(input)

    x = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=1, kernel_initializer="he_uniform", padding="same", use_bias=False)(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=3, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=1, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=3, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    return x + input

def residual_conv_block2(input, filters, skip=True):

    if skip:
        input = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, kernel_initializer="he_uniform", padding="same", use_bias=False)(input)
        input = tf.keras.layers.BatchNormalization()(input)
        input = tf.keras.layers.ReLU()(input)

    x = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=1, kernel_initializer="he_uniform", padding="same", use_bias=False)(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=3, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=1, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=3, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=1, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=3, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    return x + input

def multi_region_class(input_shape=(512, 512, 3), nclasses=2):

    h = inputs = tf.keras.Input(input_shape)

    ##########################################################################################
    h1 = h[:, 0:256, 0:256, :]
    h2 = h[:, 0:256, 256:, :]
    h3 = h[:, 256:, 0:256, :]
    h4 = h[:, 256:, 256:, :]

    h1 = residual_conv_block1(h1, 64)

    h2 = residual_conv_block1(h2, 64)
    
    h3 = residual_conv_block1(h3, 64)
    
    h4 = residual_conv_block1(h4, 64)

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", name='block1_conv1')(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", name='block1_conv2')(h)
    
    h = tf.concat([h, x], -1)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=2)(h)
    ##########################################################################################
    h1 = h[:, 0:128, 0:128, :]
    h2 = h[:, 0:128, 128:, :]
    h3 = h[:, 128:, 0:128, :]
    h4 = h[:, 128:, 128:, :]

    h1 = residual_conv_block1(h1, 128)

    h2 = residual_conv_block1(h2, 128)
    
    h3 = residual_conv_block1(h3, 128)
    
    h4 = residual_conv_block1(h4, 128)

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu", name="block2_conv1")(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu", name="block2_conv2")(h)
    h = tf.concat([h, x], -1)
    block2 = h
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=2, strides=2)(h)
    ##########################################################################################
    h1 = h[:, 0:64, 0:64, :]
    h2 = h[:, 0:64, 64:, :]
    h3 = h[:, 64:, 0:64, :]
    h4 = h[:, 64:, 64:, :]

    h1 = residual_conv_block2(h1, 256)

    h2 = residual_conv_block2(h2, 256)
    
    h3 = residual_conv_block2(h3, 256)
    
    h4 = residual_conv_block2(h4, 256)

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", name="block3_conv1")(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", name="block3_conv2")(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", name="block3_conv3")(h)
    h = tf.concat([h, x], -1)
    block3 = h
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=2, strides=2)(h)
    ##########################################################################################
    h1 = h[:, 0:32, 0:32, :]
    h2 = h[:, 0:32, 32:, :]
    h3 = h[:, 32:, 0:32, :]
    h4 = h[:, 32:, 32:, :]

    h1 = residual_conv_block2(h1, 512)

    h2 = residual_conv_block2(h2, 512)
    
    h3 = residual_conv_block2(h3, 512)
    
    h4 = residual_conv_block2(h4, 512)

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", name="block4_conv1")(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", name="block4_conv2")(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", name="block4_conv3")(h)
    h = x + h
    block4 = h
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=2, strides=2)(h)
    ##########################################################################################
    h1 = h[:, 0:16, 0:16, :]
    h2 = h[:, 0:16, 16:, :]
    h3 = h[:, 16:, 0:16, :]
    h4 = h[:, 16:, 16:, :]

    h1 = residual_conv_block2(h1, 512)

    h2 = residual_conv_block2(h2, 512)
    
    h3 = residual_conv_block2(h3, 512)
    
    h4 = residual_conv_block2(h4, 512)

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)
    tf.keras.applications.ResNet50
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", name="block5_conv1")(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", name="block5_conv2")(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", name="block5_conv3")(h)
    h = x + h
    block5 = h
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=2, strides=2)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, kernel_initializer="he_uniform", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, kernel_initializer="he_uniform", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, block5], -1)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, kernel_initializer="he_uniform", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, kernel_initializer="he_uniform", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, block4], -1)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, kernel_initializer="he_uniform", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, kernel_initializer="he_uniform", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, block3], -1)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer="he_uniform", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer="he_uniform", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([h, block2], -1)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_initializer="he_uniform", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_initializer="he_uniform", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, kernel_initializer="he_uniform", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, kernel_initializer="he_uniform", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(
        filters=2,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(h)

    model = tf.keras.Model(inputs=inputs, outputs=h)
    backbone = tf.keras.applications.VGG16(input_shape=(224,224, 3))

    model.get_layer("block1_conv1").set_weights(backbone.get_layer("block1_conv1").get_weights())
    model.get_layer("block1_conv2").set_weights(backbone.get_layer("block1_conv2").get_weights())
    model.get_layer("block2_conv1").set_weights(backbone.get_layer("block2_conv2").get_weights())
    model.get_layer("block2_conv2").set_weights(backbone.get_layer("block2_conv2").get_weights())
    model.get_layer("block3_conv1").set_weights(backbone.get_layer("block3_conv2").get_weights())
    model.get_layer("block3_conv2").set_weights(backbone.get_layer("block3_conv2").get_weights())
    model.get_layer("block3_conv3").set_weights(backbone.get_layer("block3_conv3").get_weights())
    model.get_layer("block4_conv1").set_weights(backbone.get_layer("block4_conv2").get_weights())
    model.get_layer("block4_conv2").set_weights(backbone.get_layer("block4_conv2").get_weights())
    model.get_layer("block4_conv3").set_weights(backbone.get_layer("block4_conv3").get_weights())
    model.get_layer("block5_conv1").set_weights(backbone.get_layer("block5_conv1").get_weights())
    model.get_layer("block5_conv2").set_weights(backbone.get_layer("block5_conv2").get_weights())
    model.get_layer("block5_conv3").set_weights(backbone.get_layer("block5_conv3").get_weights())
    return model
