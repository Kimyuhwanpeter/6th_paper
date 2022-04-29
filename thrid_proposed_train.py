# -*- coding:utf-8 -*-
from Cal_measurement import *
from model_ import *
from random import shuffle, random

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 448,

                           "train_txt_path": "/yuwhan/yuwhan/Dataset/Segmentation/Apple_A/train.txt",

                           "test_txt_path": "/yuwhan/yuwhan/Dataset/Segmentation/Apple_A/test.txt",
                           
                           "label_path": "/yuwhan/yuwhan/Dataset/Segmentation/Apple_A/FlowerLabels_temp/",
                           
                           "image_path": "/yuwhan/yuwhan/Dataset/Segmentation/Apple_A/FlowerImages/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "C:/Users/Yuhwan/Downloads/226/226",
                           
                           "lr": 0.0001,

                           "min_lr": 1e-7,
                           
                           "epochs": 400,

                           "total_classes": 3,

                           "ignore_label": 0,

                           "batch_size": 2,

                           "sample_images": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/6th_paper/proposed_method/Apple_A/sample_images",

                           "save_checkpoint": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/6th_paper/proposed_method/Apple_A/checkpoint",

                           "save_print": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/6th_paper/proposed_method/Apple_A/train_out.txt",

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

# @tf.function
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

    with tf.GradientTape() as tape2:

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

        #patch_1_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels_1, logits_1)
        #patch_2_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels_2, logits_2)
        #patch_3_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels_3, logits_3)
        #patch_4_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels_4, logits_4)
        #all_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, logits)

        #all_dice_loss = object_buf[1]*true_dice_loss(labels, logits[:, 1]) + object_buf[0]*false_dice_loss(labels, logits[:, 0])
        #patch_1_dice_loss = object_buf_1[1]*true_dice_loss(labels_1, logits_1[:, 1]) + object_buf_1[0]*false_dice_loss(labels_1, logits_1[:, 0])
        #patch_2_dice_loss = object_buf_2[1]*true_dice_loss(labels_2, logits_2[:, 1]) + object_buf_2[0]*false_dice_loss(labels_2, logits_2[:, 0])
        #patch_3_dice_loss = object_buf_3[1]*true_dice_loss(labels_3, logits_3[:, 1]) + object_buf_3[0]*false_dice_loss(labels_3, logits_3[:, 0])
        #patch_4_dice_loss = object_buf_4[1]*true_dice_loss(labels_4, logits_4[:, 1]) + object_buf_4[0]*false_dice_loss(labels_4, logits_4[:, 0])

        #strong_loss = all_loss + all_dice_loss
        #weak_1_loss = patch_1_dice_loss + patch_1_loss
        #weak_2_loss = patch_2_dice_loss + patch_2_loss
        #weak_3_loss = patch_3_dice_loss + patch_3_loss
        #weak_4_loss = patch_4_dice_loss + patch_4_loss

        patch_1_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels_1, logits_1)
        patch_2_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels_2, logits_2)
        patch_3_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels_3, logits_3)
        patch_4_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels_4, logits_4)
        all_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, logits)


        labels_back_indices = tf.squeeze(tf.where(labels == 0), -1)
        labels_background = tf.gather(labels, labels_back_indices)
        logits_background = tf.gather(logits[:, 0], labels_back_indices)
        labels_background = tf.where(labels_background == 0, 1, labels_background)
        labels_object_indices = tf.squeeze(tf.where(labels == 1), -1)
        labels_object = tf.gather(labels, labels_object_indices)
        logits_object = tf.gather(logits[:, 1], labels_object_indices)
        if len(labels_background) == 0:
            all_sigle_plane_loss = object_buf[1]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_object, logits_object)
        elif len(labels_object) == 0:
            all_sigle_plane_loss = object_buf[0]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_background, logits_background)
        elif len(labels_background) != 0 and len(labels_object):
            all_sigle_plane_loss = object_buf[0]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_background, logits_background) \
                + object_buf[1]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_object, logits_object)

        labels_1_back_indices = tf.squeeze(tf.where(labels_1 == 0), -1)
        labels_1_background = tf.gather(labels_1, labels_1_back_indices)
        logits_1_background = tf.gather(logits_1[:, 0], labels_1_back_indices)
        labels_1_background = tf.where(labels_1_background == 0, 1, labels_1_background)
        labels_1_object_indices = tf.squeeze(tf.where(labels_1 == 1), -1)
        labels_1_object = tf.gather(labels_1, labels_1_object_indices)
        logits_1_object = tf.gather(logits_1[:, 1], labels_1_object_indices)
        if len(labels_1_background) == 0:
            patch_1_plane_loss = object_buf_1[1]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_1_object, logits_1_object)
        elif len(labels_1_object) == 0:
            patch_1_plane_loss = object_buf_1[0]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_1_background, logits_1_background)
        elif len(labels_1_background) != 0 and len(labels_1_object):
            patch_1_plane_loss = object_buf_1[0]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_1_background, logits_1_background) \
            + object_buf_1[1]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_1_object, logits_1_object)

        labels_2_back_indices = tf.squeeze(tf.where(labels_2 == 0), -1)
        labels_2_background = tf.gather(labels_2, labels_2_back_indices)
        logits_2_background = tf.gather(logits_2[:, 0], labels_2_back_indices)
        labels_2_background = tf.where(labels_2_background == 0, 1, labels_2_background)
        labels_2_object_indices = tf.squeeze(tf.where(labels_2 == 1), -1)
        labels_2_object = tf.gather(labels_2, labels_2_object_indices)
        logits_2_object = tf.gather(logits_2[:, 1], labels_2_object_indices)
        if len(labels_2_background == 0):
            patch_2_plane_loss = object_buf_2[1]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_2_object, logits_2_object)
        elif len(labels_2_object) == 0:
            patch_2_plane_loss = object_buf_2[0]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_2_background, logits_2_background)
        elif len(labels_2_background) != 0 and len(labels_2_object):
            patch_2_plane_loss = object_buf_2[0]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_2_background, logits_2_background) \
            + object_buf_2[1]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_2_object, logits_2_object)

        labels_3_back_indices = tf.squeeze(tf.where(labels_3 == 0), -1)
        labels_3_background = tf.gather(labels_3, labels_3_back_indices)
        logits_3_background = tf.gather(logits_3[:, 0], labels_3_back_indices)
        labels_3_background = tf.where(labels_3_background == 0, 1, labels_3_background)
        labels_3_object_indices = tf.squeeze(tf.where(labels_3 == 1), -1)
        labels_3_object = tf.gather(labels_3, labels_3_object_indices)
        logits_3_object = tf.gather(logits_3[:, 1], labels_3_object_indices)
        if len(labels_3_background == 0):
            patch_3_plane_loss = object_buf_3[1]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_3_object, logits_3_object)
        elif len(labels_3_object) == 0:
            patch_3_plane_loss = object_buf_3[0]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_3_background, logits_3_background)
        elif len(labels_3_background) != 0 and len(labels_3_object):
            patch_3_plane_loss = object_buf_3[0]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_3_background, logits_3_background) \
            + object_buf_3[1]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_3_object, logits_3_object)

        labels_4_back_indices = tf.squeeze(tf.where(labels_4 == 0), -1)
        labels_4_background = tf.gather(labels_4, labels_4_back_indices)
        logits_4_background = tf.gather(logits_4[:, 0], labels_4_back_indices)
        logits_4_background = tf.where(logits_4_background == 0, 1, logits_4_background)
        labels_4_object_indices = tf.squeeze(tf.where(labels_4 == 1), -1)
        labels_4_object = tf.gather(labels_4, labels_4_object_indices)
        logits_4_object = tf.gather(logits_4[:, 1], labels_4_object_indices)
        if len(labels_4_background) == 0:
            patch_4_plane_loss = object_buf_4[1]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_4_object, logits_4_object)
        elif len(labels_4_object) == 0:
            patch_4_plane_loss = object_buf_4[0]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_4_background, logits_4_background)
        elif len(labels_4_background) != 0 and len(labels_4_object):
            patch_4_plane_loss = object_buf_4[0]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_4_background, logits_4_background) \
            + object_buf_4[1]*tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_4_object, logits_4_object)

        weak_1_loss = patch_1_loss + patch_1_plane_loss
        weak_2_loss = patch_2_loss + patch_2_plane_loss
        weak_3_loss = patch_3_loss + patch_3_plane_loss
        weak_4_loss = patch_4_loss + patch_4_plane_loss
        strong_loss = all_loss + all_sigle_plane_loss

        total_loss = weak_1_loss + weak_2_loss + weak_3_loss + weak_4_loss + strong_loss

    grads = tape2.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

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

                batch_images_1 = batch_images[:, 0:FLAGS.img_size // 2, 0:FLAGS.img_size // 2, :]   # batch_images_left_top
                batch_images_2 = batch_images[:, 0:FLAGS.img_size // 2, FLAGS.img_size // 2:, :]   # batch_images_right_top
                batch_images_3 = batch_images[:, FLAGS.img_size // 2:, 0:FLAGS.img_size // 2, :]   # batch_images_left_down
                batch_images_4 = batch_images[:, FLAGS.img_size // 2:, FLAGS.img_size // 2:, :]   # batch_images_right_down

                batch_labels_1 = batch_labels[:, 0:FLAGS.img_size // 2, 0:FLAGS.img_size // 2, :]   # batch_labels_left_top
                batch_labels_2 = batch_labels[:, 0:FLAGS.img_size // 2, FLAGS.img_size // 2:, :]   # batch_labels_right_top
                batch_labels_3 = batch_labels[:, FLAGS.img_size // 2:, 0:FLAGS.img_size // 2, :]   # batch_labels_left_down
                batch_labels_4 = batch_labels[:, FLAGS.img_size // 2:, FLAGS.img_size // 2:, :]   # batch_labels_right_down

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


                if count % 100 == 0:
                    _, _, _, _, logits = run_model(model, [batch_images_1, batch_images_2, batch_images_3, batch_images_4], False)
                    for j in range(FLAGS.batch_size):
                        label = tf.cast(batch_labels[j, :, :, 0], tf.int32).numpy()
                        object_output = tf.nn.softmax(logits[j], -1)
                        object_output = tf.argmax(object_output, -1)
                        object_output = tf.cast(object_output, tf.int32).numpy()

                        pred_mask_color = color_map[object_output]
                        label_mask_color = color_map[label]

                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, j) + "_label.png", label_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, j) + "_predict.png", pred_mask_color)


                count += 1

            tr_iter = iter(train_ge)
            iou = 0.
            cm = 0.
            f1_score_ = 0.
            recall_ = 0.
            precision_ = 0.
            for i in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                for j in range(FLAGS.batch_size):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    batch_image_1 = batch_image[:, 0:FLAGS.img_size // 2, 0:FLAGS.img_size // 2, :]   # batch_images_left_top
                    batch_image_2 = batch_image[:, 0:FLAGS.img_size // 2, FLAGS.img_size // 2:, :]   # batch_images_right_top
                    batch_image_3 = batch_image[:, FLAGS.img_size // 2:, 0:FLAGS.img_size // 2, :]   # batch_images_left_down
                    batch_image_4 = batch_image[:, FLAGS.img_size // 2:, FLAGS.img_size // 2:, :]   # batch_images_right_down

                    _, _, _, _, logits = run_model(model, [batch_image_1, batch_image_2, batch_image_3, batch_image_4], False)

                    object_output = tf.nn.softmax(logits[0], -1)
                    object_output = tf.argmax(object_output, -1)
                    object_output = tf.cast(object_output, tf.int32).numpy()
                    object_output = np.where(object_output == 1, 0, 1)

                    batch_label = tf.cast(batch_labels[j, :, :, 0], tf.uint8).numpy()
                    batch_label = np.where(batch_label == 0, 1, 0)
                    batch_label = np.array(batch_label, np.int32)

                    cm_ = Measurement(predict=object_output,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=2).MIOU()
                    
                    cm += cm_

                iou = cm[0,0]/(cm[0,0] + cm[0,1] + cm[1,0])
                precision_ = cm[0,0] / (cm[0,0] + cm[1,0])
                recall_ = cm[0,0] / (cm[0,0] + cm[0,1])
                f1_score_ = (2*precision_*recall_) / (precision_ + recall_)
            print("train mIoU = %.4f, train F1_score = %.4f, train sensitivity(recall) = %.4f, train precision = %.4f" % (iou,
                                                                                                                        f1_score_,
                                                                                                                        recall_,
                                                                                                                        precision_))

            output_text.write("Epoch: ")
            output_text.write(str(epoch))
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.write("train IoU: ")
            output_text.write("%.4f" % (iou / len(train_img_dataset)))
            output_text.write(", train F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", train sensitivity: ")
            output_text.write("%.4f" % (recall_ / len(train_img_dataset)))
            output_text.write(", train precision: ")
            output_text.write("%.4f" % (precision_ / len(train_img_dataset)))
            output_text.write("\n")

            test_iter = iter(test_ge)
            iou = 0.
            cm = 0.
            f1_score_ = 0.
            recall_ = 0.
            precision_ = 0.
            for i in range(len(test_img_dataset)):
                batch_images, batch_labels = next(test_iter)
                for j in range(1):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    batch_image_1 = batch_image[:, 0:FLAGS.img_size // 2, 0:FLAGS.img_size // 2, :]   # batch_images_left_top
                    batch_image_2 = batch_image[:, 0:FLAGS.img_size // 2, FLAGS.img_size // 2:, :]   # batch_images_right_top
                    batch_image_3 = batch_image[:, FLAGS.img_size // 2:, 0:FLAGS.img_size // 2, :]   # batch_images_left_down
                    batch_image_4 = batch_image[:, FLAGS.img_size // 2:, FLAGS.img_size // 2:, :]   # batch_images_right_down

                    _, _, _, _, logits = run_model(model, [batch_image_1, batch_image_2, batch_image_3, batch_image_4], False)

                    object_output = tf.nn.softmax(logits[0], -1)
                    object_output = tf.argmax(object_output, -1)
                    object_output = tf.cast(object_output, tf.int32).numpy()
                    object_output = np.where(object_output == 1, 0, 1)

                    batch_label = tf.cast(batch_labels[j, :, :, 0], tf.uint8).numpy()
                    batch_label = np.where(batch_label == 0, 1, 0)
                    batch_label = np.array(batch_label, np.int32)

                    cm_ = Measurement(predict=object_output,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=2).MIOU()
                    
                    cm += cm_

                iou = cm[0,0]/(cm[0,0] + cm[0,1] + cm[1,0])
                precision_ = cm[0,0] / (cm[0,0] + cm[1,0])
                recall_ = cm[0,0] / (cm[0,0] + cm[0,1])
                f1_score_ = (2*precision_*recall_) / (precision_ + recall_)


            print("test mIoU = %.4f, test F1_score = %.4f, test sensitivity(recall) = %.4f, test precision = %.4f" % (iou,
                                                                                                                    f1_score_,
                                                                                                                    recall_,
                                                                                                                    precision_))
            output_text.write("test IoU: ")
            output_text.write("%.4f" % (iou / len(test_img_dataset)))
            output_text.write(", test F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", test sensitivity: ")
            output_text.write("%.4f" % (recall_ / len(test_img_dataset)))
            output_text.write(", test precision: ")
            output_text.write("%.4f" % (precision_ / len(test_img_dataset)))
            output_text.write("\n")
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.flush()

            model_dir = "%s/%s" % (FLAGS.save_checkpoint, epoch)
            if not os.path.isdir(model_dir):
                print("Make {} folder to store the weight!".format(epoch))
                os.makedirs(model_dir)
            ckpt = tf.train.Checkpoint(model=model, optim=optim)
            ckpt_dir = model_dir + "/apple_A_model_{}.ckpt".format(epoch)
            ckpt.save(ckpt_dir)


if __name__ == "__main__":
    main()
