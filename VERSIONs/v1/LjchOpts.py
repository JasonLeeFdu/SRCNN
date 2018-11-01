from datetime import datetime
import math
import time
import tensorflow as tf
import numpy as np

def conv2d (in1, name, kh, kw, nch_in,nch_out,stride=[1,1,1,1],padding='SAME',weightDecay=0):#weightDecay 以后再加
    with tf.name_scope(name) as scope:
        #Declaration of Params
        kernel = tf.Variable(tf.truncated_normal([kh,kw,nch_in,nch_out],stddev=0.1,dtype=tf.float32),name='w')
        bias   = tf.Variable(tf.truncated_normal([nch_out],stddev=1e-3,dtype=tf.float32),name='b')
        #Define op
        out1 = tf.nn.conv2d(in1,kernel,stride,padding)
        op = tf.nn.bias_add(out1,bias)
        return op

def ReLU (in1):
    op = tf.maximum(0.0, in1)
    return op

def network(inputImg,labelImg):
    conv1 = ReLU(conv2d(inputImg,'conv1',9,9,3,64))
    conv2 = ReLU(conv2d(conv1,'conv2',1,1,64,32))
    prediction = conv2d(conv2,'resConv',5,5,32,3)
    loss = MSELoss(prediction,labelImg)
    return prediction,loss


def networkWithRes(inputImg,labelImg):
    conv1 = ReLU(conv2d(inputImg,'conv1',9,9,3,64))
    conv2 = ReLU(conv2d(conv1,'conv2',1,1,64,32))
    conv3 = conv2d(conv2,'resConv',5,5,32,3)
    prediction = conv3 + inputImg
    loss = MSELoss(prediction,labelImg)
    return prediction,loss

def MSELoss(im1,im2):
    sqRes = tf.pow((im1-im2),2)
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





