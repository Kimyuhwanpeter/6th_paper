# -*- coding:utf-8 -*-
from random import shuffle, random
from model_6 import *
from base_UNET import *

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import easydict
# 이 학습방법은 모델은 오리지널 이미지를 통해 배치를 뽑고 거기서 거기패치로 모델에 입력을 각각 시켜놓자

FLAGS = easydict.EasyDict({"img_size": 800,

                           "train_txt_path": "D:/[1]DB/[5]4th_paper_DB/Fruit/apple_pear/train.txt",

                           "test_txt_path": "D:/[1]DB/[5]4th_paper_DB/Fruit/apple_pear/test.txt",
                           
                           "tr_label_path": "D:/[1]DB/[5]4th_paper_DB/Fruit/apple_pear/FlowerLabels_temp/",
                           
                           "tr_image_path": "D:/[1]DB/[5]4th_paper_DB/Fruit/apple_pear/FlowerImages/",

                           "te_label_path": "D:/[1]DB/[5]4th_paper_DB/Fruit/apple_pear/FlowerLabels_test/",
                           
                           "te_image_path": "D:/[1]DB/[5]4th_paper_DB/Fruit/apple_pear/Flowerimages_test/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "C:/Users/Yuhwan/Downloads/398/398",
                           
                           "lr": 0.0001,

                           "min_lr": 1e-7,
                           
                           "epochs": 400,

                           "total_classes": 3,

                           "ignore_label": 0,

                           "batch_size": 1,

                           "sample_images": "C:/Users/Yuhwan/Downloads/tt/sample_images",

                           "save_checkpoint": "C:/Users/Yuhwan/Downloads/tt",

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
    #img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.cast(img, tf.float32)
    img = tf.image.random_brightness(img, max_delta=50.) 
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    # img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3], seed=123)
    no_img = img
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_jpeg(lab, 1)
    #lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # lab = tf.image.random_crop(lab, [FLAGS.img_size, FLAGS.img_size, 1], seed=123)
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

def test_func2(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)

    lab = tf.io.read_file(lab_list)
    lab = tf.image.decode_jpeg(lab, 1)

    return img, lab

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
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))
        # return (tf.keras.backend.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, images, labels):

    with tf.GradientTape() as tape:

        labels = tf.reshape(labels, [-1,])
        logits = run_model(model, images, True)
        logits = tf.reshape(logits, [-1, 2])

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def main():

    model = Unet(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), classes=2)
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!!!")

    if FLAGS.train:
        count = 0

        output_text = open(FLAGS.save_print, "w")
        
        train_list = np.loadtxt(FLAGS.train_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        train_img_dataset = [FLAGS.tr_image_path + data for data in train_list]
        test_img_dataset = [FLAGS.te_image_path + data for data in test_list]

        train_lab_dataset = [FLAGS.tr_label_path + data for data in train_list]
        test_lab_dataset = [FLAGS.te_label_path + data for data in test_list]

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

            tr_idx = len(train_img_dataset) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                batch_images = batch_images.numpy()
                batch_labels = tf.where(batch_labels > 128, 1, 0).numpy()

                #batch_labels = tf.reshape(batch_labels, [3456*51`84, ]).numpy()
                #print(np.bincount(batch_labels))

                height = batch_images.shape[1]
                width = batch_images.shape[2]
                desired_h = 1728    # Apple A
                desired_w = 1728    # Apple A
                total_loss = 0.
                for j in range(2):
                    for k in range(3):
                        split_img = batch_images[:, j*desired_h:(j+1)*desired_h, k*desired_w:(k+1)*desired_w, :]
                        split_lab = batch_labels[:, j*desired_h:(j+1)*desired_h, k*desired_w:(k+1)*desired_w, :]

                        split_img = tf.image.resize(split_img, [FLAGS.img_size, FLAGS.img_size])
                        split_lab = tf.image.resize(split_lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                        split_lab = tf.where(split_lab > 128, 1, 0)

                        class_imbal_labels_buf = 0.
                        class_imbal_labels = split_lab
                        for i in range(FLAGS.batch_size):
                            class_imbal_label = class_imbal_labels[i]
                            class_imbal_label = np.reshape(class_imbal_label, [FLAGS.img_size*FLAGS.img_size, ])
                            count_c_i_lab = np.bincount(class_imbal_label, minlength=2)
                            class_imbal_labels_buf += count_c_i_lab

                        object_buf = class_imbal_labels_buf
                        object_buf = (np.max(object_buf / np.sum(object_buf)) + 1 - (object_buf / np.sum(object_buf)))
                        object_buf = tf.nn.softmax(object_buf).numpy()

                        with tf.GradientTape() as tape:
                            split_lab = tf.reshape(split_lab, [-1,])
                            logits = run_model(model, split_img, True)
                            logits = tf.nn.softmax(tf.reshape(logits, [-1, 2]), -1)
                            
                            loss = categorical_focal_loss([object_buf[0], object_buf[1]])(tf.one_hot(split_lab, 2), logits)

                            total_loss += loss / 5

                        grads = tape.gradient(total_loss, model.trainable_variables)

                optim.apply_gradients(zip(grads, model.trainable_variables))

                if count % 10 == 0:
                    print("Epochs: {}, Loss = {} [{}/{}]".format(epoch, total_loss, step + 1, tr_idx))


                count += 1

if __name__ == "__main__":
    main()
