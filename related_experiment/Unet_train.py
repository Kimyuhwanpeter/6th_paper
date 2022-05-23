# -*- coding:utf-8 -*-
import tensorflow as tf
from base_UNET import *
from Cal_measurement import Measurement
from random import shuffle, random
# from keras_flops import get_flops
# from model_profiler import model_profiler


import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 512,

                           "train_txt_path": "/yuwhan/yuwhan/Dataset/Segmentation/Apple_A/train_fix.txt",

                           "test_txt_path": "/yuwhan/yuwhan/Dataset/Segmentation/Apple_A/test.txt",
                           
                           "tr_label_path": "/yuwhan/yuwhan/Dataset/Segmentation/Apple_A/augment_label/",
                           
                           "tr_image_path": "/yuwhan/yuwhan/Dataset/Segmentation/Apple_A/augment_train/",

                           "te_label_path": "/yuwhan/yuwhan/Dataset/Segmentation/Apple_A/FlowerLabels_test/",
                           
                           "te_image_path": "/yuwhan/yuwhan/Dataset/Segmentation/Apple_A/Flowerimages_test/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/Unet_related_5th/CWFID_low_light2/checkpoint/87",
                           
                           "lr": 0.0001,

                           "min_lr": 1e-7,
                           
                           "epochs": 200,

                           "total_classes": 2,

                           "ignore_label": 0,

                           "batch_size": 4,

                           "sample_images": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/6th_paper/related_work/Unet/sample_images",

                           "save_checkpoint": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/6th_paper/related_work/Unet/checkpoint",

                           "save_print": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/6th_paper/related_work/Unet/train_out.txt",

                           "test_images": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/Unet_related_5th/CWFID_low_light2/test_images",

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
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
     #lab = tf.image.random_crop(lab, [FLAGS.img_size, FLAGS.img_size, 1], seed=123)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)
        
    return img, no_img, lab

def test_func(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [2048, 2048])
    #img = tf.clip_by_value(img, 0, 255)
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_jpeg(lab, 1)
    # lab = tf.image.resize(lab, [2048, 2048], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)   # 이거 해체 했을 때 성능이 좋게 나왔음
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

def cal_loss(model, images, batch_labels):

    with tf.GradientTape() as tape:

        batch_labels = tf.reshape(batch_labels, [-1,])

        logits = run_model(model, images, True)
        logits = tf.reshape(logits, [-1, FLAGS.total_classes])

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(batch_labels, logits)

        # print(tf.keras.losses.BinaryCrossentropy(from_logits=True)(crop_labels, crop_logits))
        # print(tf.keras.losses.BinaryCrossentropy(from_logits=True)(weed_labels, weed_logits))
       
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss


# yilog(h(xi;θ))+(1−yi)log(1−h(xi;θ))
def main():
    tf.keras.backend.clear_session()
    # 마지막 plain은 objecttines에 대한 True or False값 즉 (mask값이고), 라벨은 annotation 이미지임 (crop/weed)
    # 학습이미지에 대해 online augmentation을 진행--> 전처리로서 필터링을 하던지 해서 , 피사체에 대한 high frequency 성분을
    # 가지고오자
    #model = PFB_model(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), OUTPUT_CHANNELS=FLAGS.total_classes-1)\
    model = Unet(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), classes=2)
    # model_pro = model_profiler(model, FLAGS.batch_size)
    # print(model_pro)

    #out = model.get_layer("activation_decoder_2_upsample").output
    #out = tf.keras.layers.Conv2D(FLAGS.total_classes-1, (1,1), name="output_layer")(out)
    #model = tf.keras.Model(inputs=model.input, outputs=out)
    
    #for layer in model.layers:
    #    if isinstance(layer, tf.keras.layers.BatchNormalization):
    #        layer.momentum = 0.9997
    #        layer.epsilon = 1e-5
        #elif isinstance(layer, tf.keras.layers.Conv2D):
        #    layer.kernel_regularizer = tf.keras.regularizers.l2(0.0005)

    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!")
    
    if FLAGS.train:
        count = 0

        output_text = open(FLAGS.save_print, "w")
        
        train_list = np.loadtxt(FLAGS.train_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        train_img_dataset = [FLAGS.tr_image_path + data for data in train_list]
        test_img_dataset = [FLAGS.te_image_path + data for data in test_list]

        train_lab_dataset = [FLAGS.tr_label_path + data for data in train_list]
        test_lab_dataset = [FLAGS.te_label_path + data for data in test_list]

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
                batch_labels = tf.where(batch_labels > 128, 1, 0).numpy()

                loss = cal_loss(model, batch_images, batch_labels)  # loss까지는 다했고 내일 test iou뽑는 부분코드 고쳐야함!
                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step+1, tr_idx, loss))

                if count % 100 == 0:
                    second_output = run_model(model, batch_images, False)
                    for i in range(FLAGS.batch_size):
                        label = tf.cast(batch_labels[i, :, :, 0], tf.int32).numpy()
                        final_output = tf.nn.softmax(second_output[i, :, :, :], -1)
                        final_output = tf.argmax(final_output, -1, output_type=tf.int32)

                        pred_mask_color = color_map[final_output]
                        label_mask_color = color_map[label]
 
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_label.png", label_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_predict.png", pred_mask_color)

                count += 1

            tr_iter = iter(train_ge)
            iou = 0.
            cm = 0.
            f1_score_ = 0.
            recall_ = 0.
            precision_ = 0.
            for i in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                batch_labels = tf.where(batch_labels > 128, 1, 0).numpy()
                for j in range(FLAGS.batch_size):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    second_output = run_model(model, batch_image, False)
                    final_output = tf.nn.softmax(second_output[0, :, :, :], -1)
                    final_output = tf.argmax(final_output, -1, output_type=tf.int32)
                    final_output = tf.where(final_output == 0, 1, 0)

                    batch_label = tf.cast(batch_labels[j, :, :, 0], tf.uint8).numpy()
                    batch_label = np.where(batch_label == 0, 1, 0)
                    batch_label = np.array(batch_label, np.int32)

                    cm_ = Measurement(predict=final_output,
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
            output_text.write("%.4f" % (iou ))
            output_text.write(", train F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", train sensitivity: ")
            output_text.write("%.4f" % (recall_ ))
            output_text.write(", train precision: ")
            output_text.write("%.4f" % (precision_ ))
            output_text.write("\n")

            test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
            test_ge = test_ge.map(test_func)
            test_ge = test_ge.batch(1)
            test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

            test_iter = iter(test_ge)
            iou = 0.
            cm = 0.
            f1_score_ = 0.
            recall_ = 0.
            precision_ = 0.
            model_ = Unet(input_shape=(FLAGS.img_size*2, FLAGS.img_size*2, 3), classes=2)
            model_.set_weights(model.get_weights())
            for i in range(len(test_img_dataset)):
                batch_images, batch_labels = next(test_iter)
                batch_labels = tf.where(batch_labels > 128, 1, 0).numpy()

                image = batch_images[0].numpy()
                # shape = tf.keras.backend.int_shape(batch_labels[0].numpy())
                height = batch_labels.shape[1]
                width = batch_labels.shape[2]

                desired_h = 1024    # Apple A   if height == 3456 and width == 5184:    # Apple A
                desired_w = 1024    # Apple A   if height == 3456 and width == 5184:    # Apple A
                for j in range(2):
                    for k in range(2):
                        split_img = image[j*desired_h:(j+1)*desired_h, k*desired_w:(k+1)*desired_w, :]
                        split_img = tf.expand_dims(split_img, 0)
                        # split_img = tf.image.resize(split_img, [FLAGS.img_size, FLAGS.img_size])
                        second_output = run_model(model_, split_img, False)
                        output = tf.nn.softmax(second_output[0, :, :, :], -1)
                        output = tf.argmax(output, -1, output_type=tf.int32)
                        output = tf.where(output == 0, 1, 0)
                        #temp_image = tf.image.resize(split_img, [1728, 1728]).numpy()
                        #temp_image = temp_image[0]

                        if j == 0 and k == 0:
                            final_output = output
                            #final_image = temp_image
                        else:
                            if j == 0:
                                final_output = tf.concat([final_output, output], 1)
                                #final_image = tf.concat([final_image, temp_image], 1)
                            else:
                                if j == 1 and k == 0:
                                    final_output2 = output
                                    #final_image2 = temp_image
                                else:
                                    final_output2 = tf.concat([final_output2, output], 1)
                                    #final_image2 = tf.concat([final_image2, temp_image], 1)

                #final_image3 = tf.concat([final_image, final_image2], 0)
                final_output = tf.concat([final_output, final_output2], 0)
                final_output = tf.expand_dims(final_output, -1)
                final_output = tf.image.resize(final_output, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                final_output = final_output[:, :, 0]
                batch_label = tf.cast(batch_labels[0, :, :, 0], tf.uint8).numpy()
                batch_label = np.where(batch_label == 0, 1, 0)
                batch_label = np.array(batch_label, np.int32)

                cm_ = Measurement(predict=final_output,
                                    label=batch_label,
                                    shape=[height*width, ],
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
            output_text.write("%.4f" % (iou))
            output_text.write(", test F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", test sensitivity: ")
            output_text.write("%.4f" % (recall_ ))
            output_text.write(", test precision: ")
            output_text.write("%.4f" % (precision_))
            output_text.write("\n")
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.flush()

            model_dir = "%s/%s" % (FLAGS.save_checkpoint, epoch)
            if not os.path.isdir(model_dir):
                print("Make {} folder to store the weight!".format(epoch))
                os.makedirs(model_dir)
            ckpt = tf.train.Checkpoint(model=model, optim=optim)
            ckpt_dir = model_dir + "/Crop_weed_model_{}.ckpt".format(epoch)
            ckpt.save(ckpt_dir)
    else:
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        test_img_dataset = [FLAGS.image_path + data for data in test_list]
        test_lab_dataset = [FLAGS.label_path + data for data in test_list]

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func2)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        test_iter = iter(test_ge)
        miou = 0.
        f1_score_ = 0.
        crop_iou = 0.
        weed_iou = 0.
        recall_ = 0.
        precision_ = 0.
        for i in range(len(test_img_dataset)):
            batch_images, nomral_img, batch_labels = next(test_iter)
            batch_labels = tf.squeeze(batch_labels, -1)
            for j in range(1):
                batch_image = tf.expand_dims(batch_images[j], 0)
                logits = run_model(model, batch_image, False) # type을 batch label과 같은 type으로 맞춰주어야함

                    
                logits = tf.nn.softmax(logits, -1)
                predict_image = tf.argmax(logits, -1)

                batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                batch_label = np.where(batch_label == 255, 0, batch_label)
                batch_label = np.where(batch_label == 128, 1, batch_label)

                miou_, crop_iou_, weed_iou_ = Measurement(predict=predict_image,
                                    label=batch_label, 
                                    shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                    total_classes=FLAGS.total_classes).MIOU()

                pred_mask_color = color_map[predict_image]  # 논문그림처럼 할것!
                batch_label = np.expand_dims(batch_label, -1)
                batch_label = np.concatenate((batch_label, batch_label, batch_label), -1)
                label_mask_color = np.zeros([FLAGS.img_size, FLAGS.img_size, 3], dtype=np.uint8)
                label_mask_color = np.where(batch_label == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), label_mask_color)
                label_mask_color = np.where(batch_label == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), label_mask_color)

                predict_image = np.expand_dims(predict_image, -1)
                temp_img = np.concatenate((predict_image, predict_image, predict_image), -1)
                image = np.concatenate((predict_image, predict_image, predict_image), -1)
                pred_mask_warping = np.where(temp_img == np.array([2,2,2], dtype=np.uint8), nomral_img[j], image)
                pred_mask_warping = np.where(temp_img == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), pred_mask_warping)
                pred_mask_warping = np.where(temp_img == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), pred_mask_warping)
                pred_mask_warping /= 255.

                name = test_img_dataset[i].split("/")[-1].split(".")[0]
                plt.imsave(FLAGS.test_images + "/" + name + "_label.png", label_mask_color)
                plt.imsave(FLAGS.test_images + "/" + name + "_predict.png", pred_mask_color)
                plt.imsave(FLAGS.test_images + "/" + name + "_predict_warp.png", pred_mask_warping[0])

                miou += miou_
                crop_iou += crop_iou_
                weed_iou += weed_iou_

        miou_ = miou[0,0]/(miou[0,0] + miou[0,1] + miou[1,0])
        crop_iou_ = crop_iou[0,0]/(crop_iou[0,0] + crop_iou[0,1] + crop_iou[1,0])
        weed_iou_ = weed_iou[0,0]/(weed_iou[0,0] + weed_iou[0,1] + weed_iou[1,0])
        recall_ = miou[0,0] / (miou[0,0] + miou[0,1])
        precision_ = miou[0,0] / (miou[0,0] + miou[1,0])
        f1_score_ = (2*precision_*recall_) / (precision_ + recall_)
        print("True positive: {}   False negative: {},\nFalse positive: {}  True negative: {}".format(miou[0,0], miou[0,1], miou[1,0], miou[1,1]))
        print("test mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), test F1_score = %.4f, test sensitivity(recall) = %.4f, test precision = %.4f" % (miou_,
                                                                                                                                            crop_iou_,
                                                                                                                                            weed_iou_,
                                                                                                                                            f1_score_,
                                                                                                                                            recall_,
                                                                                                                                            precision_))


if __name__ == "__main__":
    main()
