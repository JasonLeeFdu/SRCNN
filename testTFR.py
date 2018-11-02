import tensorflow as tf
import LjchOpts as ops
import RecordMaker as rec
import os
import time
from datetime import datetime
import numpy as np
import math
import cv2 as cv
import tools

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


TRAIN_RECORD_NAME = 'train.tfrecords'


with tf.Session() as sess:
    example = tf.train.Example()

    #train_record表示训练的tfrecords文件的路径
    record_iterator = tf.python_io.tf_record_iterator(path=TRAIN_RECORD_NAME)
    for record in record_iterator:
        example.ParseFromString(record)
        f = example.features.feature


        #解析一个example
        image_name = f['input'].bytes_list.value[0]
        img = np.fromstring(image_name, dtype=np.float32)
        img = img.reshape([32,32])

        image_name = f['label'].bytes_list.value[0]
        gt = np.fromstring(image_name, dtype=np.float32)
        gt = gt.reshape([32,32])
        a = 1;


    '''
    image_raw = f['image_raw'].bytes_list.value[0]
    xc = np.array(f['xc'].float_list.value)[:, np.newaxis]
    yc = np.array(f['yc'].float_list.value)[:, np.newaxis]
    xita = np.array(f['xita'].float_list.value)[:, np.newaxis]
    w = np.array(f['w'].float_list.value)[:, np.newaxis]
    h = np.array(f['h'].float_list.value)[:, np.newaxis]
    label = np.hstack((xc, yc, xita, w, h))

    #将label画在image上，同时打印image_name，查看三者是不是对应上的
    print(image_name.encode('utf-8'))
    img_1d = np.fromstring(image_raw, dtype=np.uint8)
    img_2d = img_1d.reshape((480, 480, -1))
    draw_bboxes(img_2d, label)

    '''




