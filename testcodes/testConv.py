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
import scipy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


TRAIN_RECORD_NAME = 'train.tfrecords'


with tf.Session() as sess:

    img = np.array([1,9,1,7,8,2,6,4,6,-1,5,2,7,3,3,8],dtype=np.float32)
    img = np.reshape(img,[1,4,4,1])
    imgTnsr = tf.placeholder(dtype=tf.float32,shape=[1,4,4,1])  #N H W C
    w1  = np.array([0,1,3,0,2,0,0,-1,0],dtype=np.float32)
    w1  = np.reshape(w1,[3,3,1,1])
    w2  = np.array([1,-1,1,-1],dtype=np.float32)
    w2  = np.reshape(w2,[2,2,1,1])
    label = tf.constant(0,dtype=tf.float32)
    conv1_w = tf.Variable(tf.constant(w1))
    conv2_w = tf.Variable(tf.constant(w2))
    conv1_b = tf.Variable(tf.constant(0.0,dtype=tf.float32))
    conv2_b = tf.Variable(tf.constant(0.0,dtype=tf.float32))


    #
    tf.global_variables_initializer().run()


    # network
    conv1c = tf.nn.conv2d(tf.constant(img),conv1_w,strides=[1,1,1,1],padding='SAME')
    conv1p = tf.nn.max_pool(conv1c,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    output = tf.nn.conv2d(conv1p,filter=conv2_w,strides=[1,1,1,1],padding='VALID')
    result = sess.run(output)
    loss   = tf.pow(label - output,2)

    grad_conv1_w,grad_conv2_w,grad_feat1,grad_feat2,grad_output = tf.gradients(loss,[conv1_w,conv2_w,conv1c,conv1p,output])

    ## get

    [grad_conv1_w,grad_conv2_w,grad_feat1,grad_feat2,grad_output] = sess.run([grad_conv1_w,grad_conv2_w,grad_feat1,grad_feat2,grad_output])


    #result  = np.reshape(result,newshape=[4,4])
    print(result)



