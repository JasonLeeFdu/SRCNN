from datetime import datetime
import math
import time
import tensorflow as tf
import numpy as np

def conv2d (in1, name, kh, kw, nch_in,nch_out,stride=[1,1,1,1],padding='SAME',weightDecay=0,varSumm=False):#weightDecay 以后再加
    with tf.name_scope(name) as scope:
        #Declaration of Params
        kernel = tf.Variable(tf.truncated_normal([kh,kw,nch_in,nch_out],stddev=1e-3,dtype=tf.float32),name='w')
        #bias   = tf.Variable(tf.truncated_normal([nch_out],stddev=1e-3,dtype=tf.float32),name='b')
        bias    = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape =[nch_out]),name='b')
        #Define op
        out1 = tf.nn.conv2d(in1,kernel,stride,padding)
        op = tf.nn.bias_add(out1,bias)
        if varSumm:
            tf.summary.histogram('w', kernel)
            tf.summary.histogram('b', bias)
        return op

def ReLU (in1):
    op = tf.nn.relu(in1)
    #op = tf.maximum(0.0, in1)
    return op

def network(inputImg,labelImg):
    with tf.name_scope('layerInput') as scope:
        conv1 = ReLU(conv2d(inputImg,'conv1',9,9,1,64,varSumm=True))
    with tf.name_scope('layerHidden') as scope:
        conv2 = ReLU(conv2d(conv1,'conv2',1,1,64,32,varSumm=True))
    with tf.name_scope('layerOutput') as scope:
        prediction = ReLU(conv2d(conv2,'resConv',5,5,32,1,varSumm=True)) #+inputImg
        loss = MSELoss(prediction,labelImg)
        tf.summary.scalar('loss',loss)
        tf.summary.image('predict_image',prediction)
    return prediction,loss

def network1(inputImg,labelImg):
    with tf.name_scope('layerInput') as scope:
        conv1 = ReLU(conv2d(inputImg,'conv1',3,3,1,64))
    with tf.name_scope('layerHidden1') as scope:
        conv2 = ReLU(conv2d(conv1,'conv2',3,3,64,128,varSumm=True))
    with tf.name_scope('layerHidden2') as scope:
        conv3 = ReLU(conv2d(conv2,'conv2',5,5,128,128,varSumm=True))
    with tf.name_scope('layerHidden3') as scope:
        conv4 = ReLU(conv2d(conv3, 'conv2', 3, 3, 128, 64, varSumm=True))
    with tf.name_scope('layerOutput') as scope:
        prediction = (conv2d(conv4,'resConv',3,3,64,1)) #+ inputImg
        loss = MSELoss(prediction,labelImg) #+ tf.reduce_mean(tf.abs(prediction-labelImg))
        tf.summary.scalar('loss',loss)
        tf.summary.image('predict_image',prediction)
    return prediction,loss


def networkWithRes(inputImg,labelImg):
    conv1 = ReLU(conv2d(inputImg,'conv1',9,9,3,64))
    conv2 = ReLU(conv2d(conv1,'conv2',1,1,64,32))
    conv3 = conv2d(conv2,'resConv',5,5,32,3)
    prediction = conv3 + inputImg
    loss = MSELoss(prediction,labelImg)
    return prediction,loss

def MSELoss(im1,im2,name='MSELoss'):
    sqRes = tf.square(im1-im2)
    loss = tf.reduce_mean(sqRes)
    return loss







'''
1.testing code
    sess = tf.Session()
    a = tf.placeholder(tf.float32,shape=[4,5,3]);
    b = a-0.5
    c = tf.maximum(b,0)
    sess.run(tf.global_variables_initializer())
    inputA = np.random.random(size=(4,5,3))
    resB = sess.run(b,feed_dict={a:inputA})
    resC = sess.run(c,feed_dict={a:inputA})
    print('------------')
    print(inputA)
    print('##############')
    print(resB, resC)
    sess.close()
    
    
    
if __name__ == '__main__':
    main()
    
    
def main():
    sess = tf.Session()
    array = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[9,10,11]]])
    #a = tf.placeholder(tf.float32, shape=[4, 5, 3])
    a = tf.constant(array)
    b = a - 0.5
    c = tf.maximum(b, 0)
    sess.run(tf.global_variables_initializer())
    inputA = np.random.random(size=(4, 5, 3))
    #resB = sess.run(b, feed_dict={a: inputA})
    #resC = sess.run(c, feed_dict={a: inputA})
    resB = sess.run(b)
    resC = sess.run(c)
    print('------------')
    print(inputA)
    print('##############')
    print(resB, resC)
    sess.close()

    
    
    

'''





