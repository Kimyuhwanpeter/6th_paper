# -*- coding:utf-8 -*-
import tensorflow as tf

def multi_region_seg_model(input_shape=(512, 512, 3)):

    h = inputs = tf.keras.Input(input_shape)

    h1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", use_bias=False)(h)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    h1_1 = tf.keras.layers.ReLU()(h1)
    h1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", use_bias=False)(h1_1)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    h1_2 = tf.keras.layers.ReLU()(h1)

    ########################################################################################################################
    h1 = tf.keras.layers.Maximum()([h1_1, h1_2])    # [512 512 64]
    h1_img = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h1) # [256 256 64]

    h1_encoder = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", use_bias=False)(h1_img)
    h1_encoder = tf.keras.layers.BatchNormalization()(h1_encoder)
    h1_encoder_1 = tf.keras.layers.ReLU()(h1_encoder) # [256 256 128]
    h1_encoder = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h1_encoder_1) # [128, 128, 128]
    h1_encoder = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", use_bias=False)(h1_encoder)
    h1_encoder = tf.keras.layers.BatchNormalization()(h1_encoder)
    h1_encoder_2 = tf.keras.layers.ReLU()(h1_encoder)   # [128, 128, 256]
    h1_encoder = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h1_encoder_2) # [64, 64, 256]
    h1_encoder = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", use_bias=False)(h1_encoder)
    h1_encoder = tf.keras.layers.BatchNormalization()(h1_encoder)
    h1_encoder_3 = tf.keras.layers.ReLU()(h1_encoder)   # [64, 64, 256]
    h1_encoder = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h1_encoder_3) # [32, 32, 256]

    h1_decoder = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same")(h1_encoder)
    h1_decoder = tf.keras.layers.BatchNormalization()(h1_decoder)
    h1_decoder = tf.keras.layers.ReLU()(h1_decoder)
    h1_decoder = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(h1_decoder)
    h1_decoder = tf.keras.layers.BatchNormalization()(h1_decoder)
    h1_decoder = tf.keras.layers.ReLU()(h1_decoder)
    h1_decoder = tf.concat([h1_decoder, h1_encoder_3], -1)  # [64, 64, 128+256]

    h1_decoder = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same")(h1_decoder)
    h1_decoder = tf.keras.layers.BatchNormalization()(h1_decoder)
    h1_decoder = tf.keras.layers.ReLU()(h1_decoder)
    h1_decoder = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(h1_decoder)
    h1_decoder = tf.keras.layers.BatchNormalization()(h1_decoder)
    h1_decoder = tf.keras.layers.ReLU()(h1_decoder)
    h1_decoder = tf.concat([h1_decoder, h1_encoder_2], -1)  # [128, 128, 64+256]

    h1_decoder = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same")(h1_decoder)
    h1_decoder = tf.keras.layers.BatchNormalization()(h1_decoder)
    h1_decoder = tf.keras.layers.ReLU()(h1_decoder)
    h1_decoder = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(h1_decoder)
    h1_decoder = tf.keras.layers.BatchNormalization()(h1_decoder)
    h1_decoder = tf.keras.layers.ReLU()(h1_decoder)
    h1_decoder = tf.concat([h1_decoder, h1_encoder_1], -1)  # [256, 256, 32+128]

    h1_decoder = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="same")(h1_decoder)
    h1_decoder = tf.keras.layers.BatchNormalization()(h1_decoder)
    h1_decoder = tf.keras.layers.ReLU()(h1_decoder)
    h1_decoder = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="same")(h1_decoder)

    h1_attention = tf.keras.layers.experimental.preprocessing.Resizing(input_shape[0] // 2, input_shape[1] // 2)(h1_decoder)
    ########################################################################################################################

    h2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", use_bias=False)(h1_img * tf.nn.sigmoid(h1_attention))
    h2 = tf.keras.layers.BatchNormalization()(h2)
    h2_1 = tf.keras.layers.ReLU()(h2)
    h2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", use_bias=False)(h2_1)
    h2 = tf.keras.layers.BatchNormalization()(h2)
    h2_2 = tf.keras.layers.ReLU()(h2)

    h2 = tf.keras.layers.Maximum()([h2_1, h2_2])
    h2_img = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h2) # [128, 128, 128]

    h2_encoder = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", use_bias=False)(h2_img)
    h2_encoder = tf.keras.layers.BatchNormalization()(h2_encoder)
    h2_encoder_1 = tf.keras.layers.ReLU()(h2_encoder) # [128 128 128]
    h2_encoder = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h2_encoder_1) # [64, 64, 128]
    h2_encoder = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", use_bias=False)(h2_encoder)
    h2_encoder = tf.keras.layers.BatchNormalization()(h2_encoder)
    h2_encoder_2 = tf.keras.layers.ReLU()(h2_encoder) 
    h2_encoder = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h2_encoder_2) # [32, 32, 256]
    h2_encoder = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", use_bias=False)(h2_encoder)
    h2_encoder = tf.keras.layers.BatchNormalization()(h2_encoder)
    h2_encoder_3 = tf.keras.layers.ReLU()(h2_encoder)   # [64, 64, 256]
    h2_encoder = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h2_encoder_3) # [16, 16, 256]

    h2_decoder = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same")(h2_encoder)
    h2_decoder = tf.keras.layers.BatchNormalization()(h2_decoder)
    h2_decoder = tf.keras.layers.ReLU()(h2_decoder)
    h2_decoder = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(h2_decoder)
    h2_decoder = tf.keras.layers.BatchNormalization()(h2_decoder)
    h2_decoder = tf.keras.layers.ReLU()(h2_decoder)
    h2_decoder = tf.concat([h2_decoder, h2_encoder_3], -1)  # [32, 32, 128+256]

    h2_decoder = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same")(h2_decoder)
    h2_decoder = tf.keras.layers.BatchNormalization()(h2_decoder)
    h2_decoder = tf.keras.layers.ReLU()(h2_decoder)
    h2_decoder = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(h2_decoder)
    h2_decoder = tf.keras.layers.BatchNormalization()(h2_decoder)
    h2_decoder = tf.keras.layers.ReLU()(h2_decoder)
    h2_decoder = tf.concat([h2_decoder, h2_encoder_2], -1)  # [64, 64, 64+256]

    h2_decoder = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same")(h2_decoder)
    h2_decoder = tf.keras.layers.BatchNormalization()(h2_decoder)
    h2_decoder = tf.keras.layers.ReLU()(h2_decoder)
    h2_decoder = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(h2_decoder)
    h2_decoder = tf.keras.layers.BatchNormalization()(h2_decoder)
    h2_decoder = tf.keras.layers.ReLU()(h2_decoder)
    h2_decoder = tf.concat([h2_decoder, h2_encoder_1], -1)  # [128, 128, 32+128]

    h2_decoder = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="same")(h2_decoder)
    h2_decoder = tf.keras.layers.BatchNormalization()(h2_decoder)
    h2_decoder = tf.keras.layers.ReLU()(h2_decoder)
    h2_decoder = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same")(h2_decoder)
    h2_decoder = tf.keras.layers.BatchNormalization()(h2_decoder)
    h2_decoder = tf.keras.layers.ReLU()(h2_decoder)
    h2_decoder = tf.concat([h2_decoder, h1_encoder_1], -1)  # [256, 256, 16+128]

    h2_decoder = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="same")(h2_decoder)
    h2_decoder = tf.keras.layers.BatchNormalization()(h2_decoder)
    h2_decoder = tf.keras.layers.ReLU()(h2_decoder)
    h2_decoder = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="same")(h2_decoder)

    h2_attention = tf.keras.layers.experimental.preprocessing.Resizing(input_shape[0] // 4, input_shape[1] // 4)(h2_decoder)
    ########################################################################################################################

    ########################################################################################################################
    h3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", use_bias=False)(h2_img * tf.nn.sigmoid(h2_attention))
    h3 = tf.keras.layers.BatchNormalization()(h3)
    h3_1 = tf.keras.layers.ReLU()(h3)
    h3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", use_bias=False)(h3_1)
    h3 = tf.keras.layers.BatchNormalization()(h3)
    h3_2 = tf.keras.layers.ReLU()(h3)

    h3 = tf.keras.layers.Maximum()([h3_1, h3_2])
    h3_img = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h3) # [64, 64, 256]

    h3_encoder = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", use_bias=False)(h3_img)
    h3_encoder = tf.keras.layers.BatchNormalization()(h3_encoder)
    h3_encoder_1 = tf.keras.layers.ReLU()(h3_encoder) # [64 64 256]
    h3_encoder = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h3_encoder) # [32, 32, 256]
    h3_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", use_bias=False)(h3_encoder)
    h3_encoder = tf.keras.layers.BatchNormalization()(h3_encoder)
    h3_encoder_2 = tf.keras.layers.ReLU()(h3_encoder) 
    h3_encoder = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h3_encoder_2) # [16, 16, 512]
    h3_encoder = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", use_bias=False)(h3_encoder)
    h3_encoder = tf.keras.layers.BatchNormalization()(h3_encoder)
    h3_encoder_3 = tf.keras.layers.ReLU()(h3_encoder) 
    h3_encoder = tf.keras.layers.MaxPool2D(pool_size=(2,2))(h3_encoder_3) # [8, 8, 512]

    h3_decoder = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding="same")(h3_encoder)
    h3_decoder = tf.keras.layers.BatchNormalization()(h3_decoder)
    h3_decoder = tf.keras.layers.ReLU()(h3_decoder)
    h3_decoder = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(h3_decoder)
    h3_decoder = tf.keras.layers.BatchNormalization()(h3_decoder)
    h3_decoder = tf.keras.layers.ReLU()(h3_decoder)
    h3_decoder = tf.concat([h3_decoder, h3_encoder_3], -1)  # [16, 16, 256+512]

    h3_decoder = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same")(h3_decoder)
    h3_decoder = tf.keras.layers.BatchNormalization()(h3_decoder)
    h3_decoder = tf.keras.layers.ReLU()(h3_decoder)
    h3_decoder = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(h3_decoder)
    h3_decoder = tf.keras.layers.BatchNormalization()(h3_decoder)
    h3_decoder = tf.keras.layers.ReLU()(h3_decoder)
    h3_decoder = tf.concat([h3_decoder, h3_encoder_2], -1)  # [32, 32, 128+512]

    h3_decoder = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same")(h3_decoder)
    h3_decoder = tf.keras.layers.BatchNormalization()(h3_decoder)
    h3_decoder = tf.keras.layers.ReLU()(h3_decoder)
    h3_decoder = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(h3_decoder)
    h3_decoder = tf.keras.layers.BatchNormalization()(h3_decoder)
    h3_decoder = tf.keras.layers.ReLU()(h3_decoder)
    h3_decoder = tf.concat([h3_decoder, h3_encoder_1], -1)  # [64, 64, 64+256]

    h3_decoder = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same")(h3_decoder)
    h3_decoder = tf.keras.layers.BatchNormalization()(h3_decoder)
    h3_decoder = tf.keras.layers.ReLU()(h3_decoder)
    h3_decoder = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(h3_decoder)
    h3_decoder = tf.keras.layers.BatchNormalization()(h3_decoder)
    h3_decoder = tf.keras.layers.ReLU()(h3_decoder)
    h3_decoder = tf.concat([h3_decoder, h2_encoder_1, h1_encoder_2], -1)  # [128, 128, 32+128+256z]

    h3_decoder = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="same")(h3_decoder)
    h3_decoder = tf.keras.layers.BatchNormalization()(h3_decoder)
    h3_decoder = tf.keras.layers.ReLU()(h3_decoder)
    h3_decoder = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same")(h3_decoder)
    h3_decoder = tf.keras.layers.BatchNormalization()(h3_decoder)
    h3_decoder = tf.keras.layers.ReLU()(h3_decoder)
    h3_decoder = tf.concat([h3_decoder, h1_encoder_1], -1)  # [256, 256, 16+128]

    h3_decoder = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="same")(h3_decoder)
    h3_decoder = tf.keras.layers.BatchNormalization()(h3_decoder)
    h3_decoder = tf.keras.layers.ReLU()(h3_decoder)
    h3_decoder = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="same")(h3_decoder)
    ########################################################################################################################
    # 라ㅣ플라시안을 h3에 넣어주자!!! 우선 실험해보고!
    return tf.keras.Model(inputs=inputs, outputs=[h1_decoder, h2_decoder, h3_decoder])
    # decoder 1 -> object segmentation; decoder 2 -> crop and weed segmentation; decoder 3 -> edge (boundary) segmentation
