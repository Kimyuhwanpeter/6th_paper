# -*- coding:utf-8 -*-
from model_ import *

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 1024,

                           "train_txt_path": "D:/[1]DB/[5]4th_paper_DB\Fruit/apple_pear/train.txt",

                           "test_txt_path": "D:/[1]DB/[5]4th_paper_DB\Fruit/apple_pear/test.txt",
                           
                           "label_path": "D:/[1]DB/[5]4th_paper_DB\Fruit/apple_pear/FlowerLabels_temp/",
                           
                           "image_path": "D:/[1]DB/[5]4th_paper_DB\Fruit/apple_pear/FlowerImages/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "C:/Users/Yuhwan/Downloads/226/226",
                           
                           "lr": 0.0001,

                           "min_lr": 1e-7,
                           
                           "epochs": 400,

                           "total_classes": 3,

                           "ignore_label": 0,

                           "batch_size": 2,

                           "sample_images": "C:/Users/Yuhwan/Downloads/sample_images",

                           "save_checkpoint": "/content/drive/MyDrive/6th_paper/checkpoint",

                           "save_print": "C:/Users/Yuhwan/Downloads/train_out.txt",

                           "train_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/train_loss.txt",

                           "train_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/train_acc.txt",

                           "val_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/val_loss.txt",

                           "val_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/val_acc.txt",

                           "test_images": "C:/Users/Yuhwan/Downloads/test_images",

                           "train": True})

optim = tf.keras.optimizers.Adam(FLAGS.lr)
color_map = np.array([[0, 0, 0],[255,0,0]], np.uint8)
def tr_func(image_list, label_list):

    h = tf.random.uniform([1], 1e-2, 30)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, 30)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.cast(img, tf.float32)
    #img = tf.image.random_brightness(img, max_delta=50.) 
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    no_img = img
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_jpeg(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)
        
    return img, no_img, lab

def test_func(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.clip_by_value(img, 0, 255)
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_jpeg(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

def true_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def false_dice_loss(y_true, y_pred):
    y_true = 1 - tf.cast(y_true, tf.float32)
    y_pred = 1 - tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -tf.keras.backend.mean((alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
               + ((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0)))
        # return -tf.keras.backend.sum(alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
        #        -tf.keras.backend.sum((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0))

    return binary_focal_loss_fixed

def two_region_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2*(tf.reduce_sum(y_true*y_pred) + tf.reduce_sum((1 - y_true)*(1 - y_pred)))
    denominator = tf.reduce_sum(y_true + y_pred) + tf.reduce_sum(2 - y_true - y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ??  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)
    alpha = np.reshape(alpha, [1, 2])

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        #y_pred = tf.experimental.numpy.clip(y_pred, epsilon, 1. - epsilon)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))
        # return (tf.keras.backend.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -tf.keras.backend.mean((alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
               + ((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0)))
        # return -tf.keras.backend.sum(alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
        #        -tf.keras.backend.sum((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0))

    return binary_focal_loss_fixed

#@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)


def cal_loss(model, 
            images_1, 
            images_2, 
            images_3, 
            images_4,
            batch_labels_1, 
            batch_labels_2,
            batch_labels_3, 
            batch_labels_4, 
            batch_labels,
            object_buf_1,
            object_buf_2,
            object_buf_3,
            object_buf_4,
            object_buf):

    with tf.GradientTape(persistent=True) as tape2:

        labels_1 = tf.reshape(batch_labels_1, [-1,])
        labels_2 = tf.reshape(batch_labels_2, [-1,])
        labels_3 = tf.reshape(batch_labels_3, [-1,])
        labels_4 = tf.reshape(batch_labels_4, [-1,])
        labels = tf.reshape(batch_labels, [-1,])
        logits_1, logits_2, logits_3, logits_4, logits = run_model(model, [images_1, images_2, images_3, images_4], True)
        logits_1 = tf.reshape(logits_1, [-1, 2])
        logits_2 = tf.reshape(logits_2, [-1, 2])
        logits_3 = tf.reshape(logits_3, [-1, 2])
        logits_4 = tf.reshape(logits_4, [-1, 2])
        logits = tf.reshape(logits, [-1, 2])

        # 일단 focal loss만 사용해보자 각 패치에 대한 object 개수 조건문은 내일 달아주고!!! 지금은 일단 러프하게코딩
        patch_1_loss = categorical_focal_loss(alpha=[object_buf_1[0], object_buf_1[1]])(tf.one_hot(labels_1, 2), 
                                                                                    tf.nn.softmax(logits_1, -1))
        patch_2_loss = categorical_focal_loss(alpha=[object_buf_2[0], object_buf_2[1]])(tf.one_hot(labels_2, 2), 
                                                                                    tf.nn.softmax(logits_2, -1))
        patch_3_loss = categorical_focal_loss(alpha=[object_buf_3[0], object_buf_3[1]])(tf.one_hot(labels_3, 2), 
                                                                                    tf.nn.softmax(logits_3, -1))
        patch_4_loss = categorical_focal_loss(alpha=[object_buf_4[0], object_buf_4[1]])(tf.one_hot(labels_4, 2), 
                                                                                    tf.nn.softmax(logits_4, -1))
        all_loss = categorical_focal_loss(alpha=[object_buf[0], object_buf[1]])(tf.one_hot(labels, 2), 
                                                                            tf.nn.softmax(logits, -1))

        all_dice_loss = object_buf[1]*true_dice_loss(labels, logits[:, 1]) + object_buf[0]*false_dice_loss(labels, logits[:, 0])
        patch_1_dice_loss = object_buf_1[1]*true_dice_loss(labels_1, logits_1[:, 1]) + object_buf_1[0]*false_dice_loss(labels_1, logits_1[:, 0])
        patch_2_dice_loss = object_buf_2[1]*true_dice_loss(labels_2, logits_2[:, 1]) + object_buf_2[0]*false_dice_loss(labels_2, logits_2[:, 0])
        patch_3_dice_loss = object_buf_3[1]*true_dice_loss(labels_3, logits_3[:, 1]) + object_buf_3[0]*false_dice_loss(labels_3, logits_3[:, 0])
        patch_4_dice_loss = object_buf_4[1]*true_dice_loss(labels_4, logits_4[:, 1]) + object_buf_4[0]*false_dice_loss(labels_4, logits_4[:, 0])

        strong_loss = all_loss + all_dice_loss
        weak_1_loss = patch_1_dice_loss + patch_1_loss
        weak_2_loss = patch_2_dice_loss + patch_2_loss
        weak_3_loss = patch_3_dice_loss + patch_3_loss
        weak_4_loss = patch_4_dice_loss + patch_4_loss

    grads_weak_1 = tape2.gradient(weak_1_loss, model.trainable_variables)
    grads_weak_2 = tape2.gradient(weak_2_loss, model.trainable_variables)
    grads_weak_3 = tape2.gradient(weak_3_loss, model.trainable_variables)
    grads_weak_4 = tape2.gradient(weak_4_loss, model.trainable_variables)
    grads_strong = tape2.gradient(strong_loss, model.trainable_variables)

    optim.apply_gradients(zip(grads_weak_1, model.trainable_variables))
    optim.apply_gradients(zip(grads_weak_2, model.trainable_variables))
    optim.apply_gradients(zip(grads_weak_3, model.trainable_variables))
    optim.apply_gradients(zip(grads_weak_4, model.trainable_variables))
    optim.apply_gradients(zip(grads_strong, model.trainable_variables))

    return total_loss

def main():

    model = patch_model(input_shape=(FLAGS.img_size // 2, FLAGS.img_size // 2, 3), classes=2)
    model.summary()


    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!!!")

    if FLAGS.train:
        count = 0;

        output_text = open(FLAGS.save_print, "w")
        
        train_list = np.loadtxt(FLAGS.train_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        train_img_dataset = [FLAGS.image_path + data for data in train_list]
        test_img_dataset = [FLAGS.image_path + data for data in test_list]

        train_lab_dataset = [FLAGS.label_path + data for data in train_list]
        test_lab_dataset = [FLAGS.label_path + data for data in test_list]

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img_dataset, train_lab_dataset))
            shuffle(A)
            train_img_dataset, train_lab_dataset = zip(*A)
            train_img_dataset, train_lab_dataset = np.array(train_img_dataset), np.array(train_lab_dataset)

            train_ge = tf.data.Dataset.from_tensor_slices((train_img_dataset, train_lab_dataset))
            train_ge = train_ge.shuffle(len(train_img_dataset))
            train_ge = train_ge.map(tr_func)
            train_ge = train_ge.batch(FLAGS.batch_size)
            train_ge = train_ge.prefetch(tf.data.experimental.AUTOTUNE)
            tr_iter = iter(train_ge)

            tr_iter = iter(train_ge)
            tr_idx = len(train_img_dataset) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, print_images, batch_labels = next(tr_iter)

                batch_images_1 = batch_images[:, 0:512, 0:512, :]   # batch_images_left_top
                batch_images_2 = batch_images[:, 0:512, 512:, :]   # batch_images_right_top
                batch_images_3 = batch_images[:, 512:, 0:512, :]   # batch_images_left_down
                batch_images_4 = batch_images[:, 512:, 512:, :]   # batch_images_right_down

                batch_labels_1 = batch_labels[:, 0:512, 0:512, :]   # batch_labels_left_top
                batch_labels_2 = batch_labels[:, 0:512, 512:, :]   # batch_labels_right_top
                batch_labels_3 = batch_labels[:, 512:, 0:512, :]   # batch_labels_left_down
                batch_labels_4 = batch_labels[:, 512:, 512:, :]   # batch_labels_right_down

                batch_labels_1 = batch_labels_1.numpy()
                batch_labels_1 = np.where(batch_labels_1 == 255, 1, 0)
                batch_labels_2 = batch_labels_2.numpy()
                batch_labels_2 = np.where(batch_labels_2 == 255, 1, 0)
                batch_labels_3 = batch_labels_3.numpy()
                batch_labels_3 = np.where(batch_labels_3 == 255, 1, 0)
                batch_labels_4 = batch_labels_4.numpy()
                batch_labels_4 = np.where(batch_labels_4 == 255, 1, 0)
                batch_labels = batch_labels.numpy()
                batch_labels = np.where(batch_labels == 255, 1, 0)

                ###############################################################################################
                # total batch labels에 관한 object buf
                class_imbal_labels_buf = 0.
                class_imbal_labels = batch_labels
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels[i]
                    class_imbal_label = np.reshape(class_imbal_label, [FLAGS.img_size*FLAGS.img_size, ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=2)
                    class_imbal_labels_buf += count_c_i_lab

                object_buf = class_imbal_labels_buf
                object_buf = (np.max(object_buf / np.sum(object_buf)) + 1 - (object_buf / np.sum(object_buf)))
                object_buf = tf.nn.softmax(object_buf).numpy()
                ###############################################################################################

                ###############################################################################################
                # batch_labels_1's object buf
                class_imbal_labels_buf_1 = 0.
                class_imbal_labels_1 = batch_labels_1
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels_1[i]
                    class_imbal_label = np.reshape(class_imbal_label, [(FLAGS.img_size // 2) * (FLAGS.img_size // 2), ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=2)
                    class_imbal_labels_buf_1 += count_c_i_lab

                object_buf_1 = class_imbal_labels_buf_1
                object_buf_1 = (np.max(object_buf_1 / np.sum(object_buf_1)) + 1 - (object_buf_1 / np.sum(object_buf_1)))
                object_buf_1 = tf.nn.softmax(object_buf_1).numpy()
                ###############################################################################################

                ###############################################################################################
                # batch_labels_2's object buf
                class_imbal_labels_buf_2 = 0.
                class_imbal_labels_2 = batch_labels_2
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels_2[i]
                    class_imbal_label = np.reshape(class_imbal_label, [(FLAGS.img_size // 2) * (FLAGS.img_size // 2), ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=2)
                    class_imbal_labels_buf_2 += count_c_i_lab

                object_buf_2 = class_imbal_labels_buf_2
                object_buf_2 = (np.max(object_buf_2 / np.sum(object_buf_2)) + 1 - (object_buf_2 / np.sum(object_buf_2)))
                object_buf_2 = tf.nn.softmax(object_buf_2).numpy()
                ###############################################################################################

                ###############################################################################################
                # batch_labels_3's object buf
                class_imbal_labels_buf_3 = 0.
                class_imbal_labels_3 = batch_labels_3
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels_3[i]
                    class_imbal_label = np.reshape(class_imbal_label, [(FLAGS.img_size // 2) * (FLAGS.img_size // 2), ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=2)
                    class_imbal_labels_buf_3 += count_c_i_lab

                object_buf_3 = class_imbal_labels_buf_3
                object_buf_3 = (np.max(object_buf_3 / np.sum(object_buf_3)) + 1 - (object_buf_3 / np.sum(object_buf_3)))
                object_buf_3 = tf.nn.softmax(object_buf_3).numpy()
                ###############################################################################################

                ###############################################################################################
                # batch_labels_4's object buf
                class_imbal_labels_buf_4 = 0.
                class_imbal_labels_4 = batch_labels_4
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels_4[i]
                    class_imbal_label = np.reshape(class_imbal_label, [(FLAGS.img_size // 2) * (FLAGS.img_size // 2), ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=2)
                    class_imbal_labels_buf_4 += count_c_i_lab

                object_buf_4 = class_imbal_labels_buf_4
                object_buf_4 = (np.max(object_buf_4 / np.sum(object_buf_4)) + 1 - (object_buf_4 / np.sum(object_buf_4)))
                object_buf_4 = tf.nn.softmax(object_buf_4).numpy()
                ###############################################################################################

                loss = cal_loss(model, 
                                batch_images_1, 
                                batch_images_2, 
                                batch_images_3, 
                                batch_images_4,
                                batch_labels_1, 
                                batch_labels_2,
                                batch_labels_3, 
                                batch_labels_4, 
                                batch_labels,
                                object_buf_1,
                                object_buf_2,
                                object_buf_3,
                                object_buf_4,
                                object_buf)

                if count % 10 == 0:
                    print("Epochs: {}, Loss = {} [{}/{}]".format(epoch, loss, step + 1, tr_idx))




if __name__ == "__main__":
    main()
