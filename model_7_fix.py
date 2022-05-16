# -*- coding:utf-8 -*-
import tensorflow as tf

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

def multi_scale_network_1(input_shape=(512, 512, 3), nclasses=1):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv1")(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv2")(h)
    
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2, padding="same")(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu", name="conv3")(h)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu", name="conv4")(h)
    block_1 = h

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=2, padding="same")(h)
    downsample_1 = h
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", name="conv5")(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", name="conv6")(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", name="conv7")(h)
    block_2 = h

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=2, strides=2, padding="same")(h)
    downsample_2 = h
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", name="conv8")(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", name="conv9")(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", name="conv10")(h)
    block_3 = h

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=2, strides=2, padding="same")(h)
    downsample_3= h
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", name="conv11")(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", name="conv12")(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", name="conv13")(h)
    block_4 = h

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=2, strides=2, padding="same")(h)
    downsample_4 = h
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)    

    h = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2)(h)
    h = attention_module(downsample_4, h, 2)
    output_4 = h
    output_4 = tf.image.resize(output_4, [input_shape[0], input_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output_4 = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv4',
    )(output_4)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_4, h], -1)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)    

    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2)(h)
    h = attention_module(downsample_3, h, 4)
    output_3 = h
    output_3 = tf.image.resize(output_3, [input_shape[0], input_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output_3 = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv3',
    )(output_3 * tf.nn.sigmoid(output_4))
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_3, h], -1)    

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2)(h)
    h = attention_module(downsample_2, h, 4)
    output_2 = h
    output_2 = tf.image.resize(output_2, [input_shape[0], input_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output_2 = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv2',
    )(output_2 * tf.nn.sigmoid(output_3))
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_2, h], -1)    

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2)(h)
    h = attention_module(downsample_1, h, 4)
    output_1 = h
    output_1 = tf.image.resize(output_1, [input_shape[0], input_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output_1 = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv1',
    )(output_1 * tf.nn.sigmoid(output_2))
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_1, h], -1)    

    h = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=2, strides=2)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    output = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(h * tf.nn.sigmoid(output_1))

    model = tf.keras.Model(inputs=inputs, outputs=[output_4, output_3, output_2, output_1, output])

    backbone = tf.keras.applications.VGG16(input_shape=(224, 224, 3))

    model.get_layer("conv1").set_weights(backbone.get_layer("block1_conv1").get_weights())
    model.get_layer("conv2").set_weights(backbone.get_layer("block1_conv2").get_weights())
    model.get_layer("conv3").set_weights(backbone.get_layer("block2_conv1").get_weights())
    model.get_layer("conv4").set_weights(backbone.get_layer("block2_conv2").get_weights())
    model.get_layer("conv5").set_weights(backbone.get_layer("block3_conv1").get_weights())
    model.get_layer("conv6").set_weights(backbone.get_layer("block3_conv2").get_weights())
    model.get_layer("conv7").set_weights(backbone.get_layer("block3_conv3").get_weights())
    model.get_layer("conv8").set_weights(backbone.get_layer("block4_conv1").get_weights())
    model.get_layer("conv9").set_weights(backbone.get_layer("block4_conv2").get_weights())
    model.get_layer("conv10").set_weights(backbone.get_layer("block4_conv3").get_weights())
    model.get_layer("conv11").set_weights(backbone.get_layer("block5_conv1").get_weights())
    model.get_layer("conv12").set_weights(backbone.get_layer("block5_conv2").get_weights())
    model.get_layer("conv13").set_weights(backbone.get_layer("block5_conv3").get_weights())

    return model

def multi_scale_network_2(input_shape=(512, 512, 3), nclasses=2):
    
    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv1")(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv2")(h)
    
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2, padding="same")(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu", name="conv3")(h)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu", name="conv4")(h)
    block_1 = h

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=2, padding="same")(h)
    downsample_1 = h
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", name="conv5")(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", name="conv6")(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", name="conv7")(h)
    block_2 = h

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=2, strides=2, padding="same")(h)
    downsample_2 = h
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", name="conv8")(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", name="conv9")(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", name="conv10")(h)
    block_3 = h

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=2, strides=2, padding="same")(h)
    downsample_3= h
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", name="conv11")(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", name="conv12")(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", name="conv13")(h)
    block_4 = h

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=2, strides=2, padding="same")(h)
    downsample_4 = h
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)    

    h = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2)(h)
    h = attention_module(downsample_4, h, 2)
    output_4 = h
    output_4 = tf.image.resize(output_4, [input_shape[0], input_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output_4 = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv4',
    )(output_4)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_4, h], -1)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)    

    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2)(h)
    h = attention_module(downsample_3, h, 4)
    output_3 = h
    output_3 = tf.image.resize(output_3, [input_shape[0], input_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output_3 = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv3',
    )(output_3)
    output_3 = output_3 * tf.nn.softmax(output_4, -1)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_3, h], -1)    

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2)(h)
    h = attention_module(downsample_2, h, 4)
    output_2 = h
    output_2 = tf.image.resize(output_2, [input_shape[0], input_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output_2 = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv2',
    )(output_2)
    output_2 = output_2 * tf.nn.softmax(output_3, -1)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_2, h], -1)    

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2)(h)
    h = attention_module(downsample_1, h, 4)
    output_1 = h
    output_1 = tf.image.resize(output_1, [input_shape[0], input_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output_1 = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv1',
    )(output_1)
    output_1 = output_1 * tf.nn.softmax(output_2, -1)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_1, h], -1)    

    h = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=2, strides=2)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    output = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(h)
    output = output * tf.nn.softmax(output_1, -1)

    model = tf.keras.Model(inputs=inputs, outputs=[output_4, output_3, output_2, output_1, output])

    backbone = tf.keras.applications.VGG16(input_shape=(224, 224, 3))

    model.get_layer("conv1").set_weights(backbone.get_layer("block1_conv1").get_weights())
    model.get_layer("conv2").set_weights(backbone.get_layer("block1_conv2").get_weights())
    model.get_layer("conv3").set_weights(backbone.get_layer("block2_conv1").get_weights())
    model.get_layer("conv4").set_weights(backbone.get_layer("block2_conv2").get_weights())
    model.get_layer("conv5").set_weights(backbone.get_layer("block3_conv1").get_weights())
    model.get_layer("conv6").set_weights(backbone.get_layer("block3_conv2").get_weights())
    model.get_layer("conv7").set_weights(backbone.get_layer("block3_conv3").get_weights())
    model.get_layer("conv8").set_weights(backbone.get_layer("block4_conv1").get_weights())
    model.get_layer("conv9").set_weights(backbone.get_layer("block4_conv2").get_weights())
    model.get_layer("conv10").set_weights(backbone.get_layer("block4_conv3").get_weights())
    model.get_layer("conv11").set_weights(backbone.get_layer("block5_conv1").get_weights())
    model.get_layer("conv12").set_weights(backbone.get_layer("block5_conv2").get_weights())
    model.get_layer("conv13").set_weights(backbone.get_layer("block5_conv3").get_weights())

    return model
