# -*- coding:utf-8 -*-
import tensorflow as tf

def multi_region_class(input_shape=(512, 512, 3), nclasses=2):

    h = inputs = tf.keras.Input(input_shape)

    ##########################################################################################
    h1 = h[:, 0:256, 0:256, :]
    h2 = h[:, 0:256, 256:, :]
    h3 = h[:, 256:, 0:256, :]
    h4 = h[:, 256:, 256:, :]

    h1 = tf.pad(h1, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, name='conv1_1', use_bias=True)(h1)
    h1 = tf.keras.layers.BatchNormalization()(h1)

    h2 = tf.pad(h2, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, name='conv1_2', use_bias=True)(h2)
    h2 = tf.keras.layers.BatchNormalization()(h2)
    
    h3 = tf.pad(h3, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, name='conv1_3', use_bias=True)(h3)
    h3 = tf.keras.layers.BatchNormalization()(h3)
    
    h4 = tf.pad(h4, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, name='conv1_4', use_bias=True)(h4)
    h4 = tf.keras.layers.BatchNormalization()(h4)
    

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", name='conv2')(h)
    h = tf.nn.sigmoid(x) * h
    h = tf.nn.relu(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", name='conv3')(h)
    h = tf.nn.relu(x) + h
    block_1 = h
    ##########################################################################################
    h = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h)
    ##########################################################################################
    h1 = h[:, 0:128, 0:128, :]
    h2 = h[:, 0:128, 128:, :]
    h3 = h[:, 128:, 0:128, :]
    h4 = h[:, 128:, 128:, :]

    h1 = tf.pad(h1, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, name='conv4_1', use_bias=True)(h1)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    
    h2 = tf.pad(h2, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, name='conv4_2', use_bias=True)(h2)
    h2 = tf.keras.layers.BatchNormalization()(h2)
    
    h3 = tf.pad(h3, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, name='conv4_3', use_bias=True)(h3)
    h3 = tf.keras.layers.BatchNormalization()(h3)
    
    h4 = tf.pad(h4, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, name='conv4_4', use_bias=True)(h4)
    h4 = tf.keras.layers.BatchNormalization()(h4)
    

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu", name='conv5')(h)
    h = tf.nn.sigmoid(x) * h
    h = tf.nn.relu(h)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu", name='conv6')(h)
    h = tf.nn.relu(x) + h
    block_2 = h
    ##########################################################################################
    h = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h)
    ##########################################################################################
    h1 = h[:, 0:64, 0:64, :]
    h2 = h[:, 0:64, 64:, :]
    h3 = h[:, 64:, 0:64, :]
    h4 = h[:, 64:, 64:, :]

    h1 = tf.pad(h1, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, name='conv7_1', groups=8, use_bias=True)(h1)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    
    h2 = tf.pad(h2, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, name='conv7_2', groups=8, use_bias=True)(h2)
    h2 = tf.keras.layers.BatchNormalization()(h2)
    
    h3 = tf.pad(h3, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, name='conv7_3', groups=8, use_bias=True)(h3)
    h3 = tf.keras.layers.BatchNormalization()(h3)
    
    h4 = tf.pad(h4, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, name='conv7_4', groups=8, use_bias=True)(h4)
    h4 = tf.keras.layers.BatchNormalization()(h4)
    

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", name='conv8')(h)
    h = tf.nn.sigmoid(x) * h
    h = tf.nn.relu(h)
    
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", name='conv9')(h)
    h = tf.nn.sigmoid(x) * h
    h = tf.nn.relu(h)
    
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", name='conv10')(h)
    h = tf.nn.relu(x) + h
    block_3 = h
    ##########################################################################################
    h = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h)
    ##########################################################################################
    h1 = h[:, 0:32, 0:32, :]
    h2 = h[:, 0:32, 32:, :]
    h3 = h[:, 32:, 0:32, :]
    h4 = h[:, 32:, 32:, :]

    h1 = tf.pad(h1, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, name='conv11_1', groups=16, use_bias=True)(h1)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    
    h2 = tf.pad(h2, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, name='conv11_2', groups=16, use_bias=True)(h2)
    h2 = tf.keras.layers.BatchNormalization()(h2)
    
    h3 = tf.pad(h3, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, name='conv11_3', groups=16, use_bias=True)(h3)
    h3 = tf.keras.layers.BatchNormalization()(h3)
    
    h4 = tf.pad(h4, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, name='conv11_4', groups=16, use_bias=True)(h4)
    h4 = tf.keras.layers.BatchNormalization()(h4)
    

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", name='conv12')(h)
    h = tf.nn.sigmoid(x) * h
    h = tf.nn.relu(h)
    
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", name='conv13')(h)
    h = tf.nn.sigmoid(x) * h
    h = tf.nn.relu(h)
    
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", name='conv14')(h)
    h = tf.nn.relu(x) + h
    block_4 = h
    ##########################################################################################
    h = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h)
    ##########################################################################################
    h1 = h[:, 0:16, 0:16, :]
    h2 = h[:, 0:16, 16:, :]
    h3 = h[:, 16:, 0:16, :]
    h4 = h[:, 16:, 16:, :]

    h1 = tf.pad(h1, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, name='conv15_1', groups=16, use_bias=True)(h1)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    
    h2 = tf.pad(h2, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, name='conv15_2', groups=16, use_bias=True)(h2)
    h2 = tf.keras.layers.BatchNormalization()(h2)
    
    h3 = tf.pad(h3, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, name='conv15_3', groups=16, use_bias=True)(h3)
    h3 = tf.keras.layers.BatchNormalization()(h3)
    
    h4 = tf.pad(h4, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, name='conv15_4', groups=16, use_bias=True)(h4)
    h4 = tf.keras.layers.BatchNormalization()(h4)
    

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)    

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", name='conv16')(h)
    h = tf.nn.sigmoid(x) * h
    h = tf.nn.relu(h)
    
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", name='conv17')(h)
    h = tf.nn.sigmoid(x) * h
    h = tf.nn.relu(h)
    
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", name='conv18')(h)
    h = tf.nn.relu(x) + h
    ##########################################################################################
    ##########################################################################################
    # decoder
    h1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding="same", use_bias=False)(h1)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    
    h2 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding="same", use_bias=False)(h2)
    h2 = tf.keras.layers.BatchNormalization()(h2)
    
    h3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding="same", use_bias=False)(h3)
    h3 = tf.keras.layers.BatchNormalization()(h3)
    
    h4 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding="same", use_bias=False)(h4)
    h4 = tf.keras.layers.BatchNormalization()(h4)
    

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)

    h = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.nn.sigmoid(x) * h
    h = tf.keras.layers.ReLU()(h)

    h = tf.concat([block_4, h], -1)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.nn.relu(x) + h
    
    h1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False)(h1)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    
    h2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False)(h2)
    h2 = tf.keras.layers.BatchNormalization()(h2)
    
    h3 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False)(h3)
    h3 = tf.keras.layers.BatchNormalization()(h3)
    
    h4 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False)(h4)
    h4 = tf.keras.layers.BatchNormalization()(h4)
    

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)

    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.nn.sigmoid(x) * h
    h = tf.keras.layers.ReLU()(h)

    h = tf.concat([block_3, h], -1)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.nn.relu(x) + h

    h1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False)(h1)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    
    h2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False)(h2)
    h2 = tf.keras.layers.BatchNormalization()(h2)
    
    h3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False)(h3)
    h3 = tf.keras.layers.BatchNormalization()(h3)
    
    h4 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False)(h4)
    h4 = tf.keras.layers.BatchNormalization()(h4)
    

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)

    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.nn.sigmoid(x) * h
    h = tf.keras.layers.ReLU()(h)

    h = tf.concat([block_2, h], -1)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.nn.relu(x) + h

    h1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same", use_bias=False)(h1)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    
    h2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same", use_bias=False)(h2)
    h2 = tf.keras.layers.BatchNormalization()(h2)
    
    h3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same", use_bias=False)(h3)
    h3 = tf.keras.layers.BatchNormalization()(h3)
    
    h4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same", use_bias=False)(h4)
    h4 = tf.keras.layers.BatchNormalization()(h4)
    

    h1h2 = tf.concat([h1, h2], 2)
    h3h4 = tf.concat([h3, h4], 2)
    x = tf.concat([h1h2, h3h4], 1)

    h = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.nn.sigmoid(x) * h
    h = tf.keras.layers.ReLU()(h)

    h = tf.concat([block_1, h], -1)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=32, kernel_size=3, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.nn.relu(x) + h

    output = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(h)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    backbone = tf.keras.applications.VGG16(input_shape=(224, 224, 3))

    model.get_layer('conv2').set_weights(backbone.get_layer('block1_conv1').get_weights())
    model.get_layer('conv3').set_weights(backbone.get_layer('block1_conv2').get_weights())
    model.get_layer('conv5').set_weights(backbone.get_layer('block2_conv1').get_weights())
    model.get_layer('conv6').set_weights(backbone.get_layer('block2_conv2').get_weights())
    model.get_layer('conv8').set_weights(backbone.get_layer('block3_conv1').get_weights())
    model.get_layer('conv9').set_weights(backbone.get_layer('block3_conv2').get_weights())
    model.get_layer('conv10').set_weights(backbone.get_layer('block3_conv3').get_weights())
    model.get_layer('conv12').set_weights(backbone.get_layer('block4_conv1').get_weights())
    model.get_layer('conv13').set_weights(backbone.get_layer('block4_conv2').get_weights())
    model.get_layer('conv14').set_weights(backbone.get_layer('block4_conv3').get_weights())
    model.get_layer('conv16').set_weights(backbone.get_layer('block5_conv1').get_weights())
    model.get_layer('conv17').set_weights(backbone.get_layer('block5_conv2').get_weights())
    model.get_layer('conv18').set_weights(backbone.get_layer('block5_conv3').get_weights())

    model.get_layer('conv1_1').set_weights(backbone.get_layer('block1_conv1').get_weights())
    model.get_layer('conv1_2').set_weights(backbone.get_layer('block1_conv1').get_weights())
    model.get_layer('conv1_3').set_weights(backbone.get_layer('block1_conv1').get_weights())
    model.get_layer('conv1_4').set_weights(backbone.get_layer('block1_conv1').get_weights())

    model.get_layer('conv4_1').set_weights(backbone.get_layer('block2_conv1').get_weights())
    model.get_layer('conv4_2').set_weights(backbone.get_layer('block2_conv1').get_weights())
    model.get_layer('conv4_3').set_weights(backbone.get_layer('block2_conv1').get_weights())
    model.get_layer('conv4_4').set_weights(backbone.get_layer('block2_conv1').get_weights())

    return tf.keras.Model(inputs=inputs, outputs=output)
