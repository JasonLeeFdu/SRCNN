# Good cookbook
# https://blog.csdn.net/dou3516/article/details/81102209
# https://pytorch.org/docs/stable/index.html

# Code in file nn/two_layer_net_nn.py
import torch
import torchvision

import scipy.io as scio
import os
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import scipy


def SRCNN():
    net = torch.nn.Sequential()
    data = scio.loadmat(prettrainModelFile)
    c1_w = data['weights_conv1']
    c1_w = np.transpose(c1_w)
    c1_w = np.expand_dims(c1_w,axis=1)
    c1_w = np.reshape(c1_w,newshape=[64,1,9,9])                         # if we need exchange of the last dims latter on
    c1_w = torch.nn.Parameter(torch.Tensor(c1_w))
    c1_h = data['biases_conv1']
    c1_h = c1_h.reshape([-1])                                           # if we need exchange of the last dims latter on
    c1_h = torch.nn.Parameter(torch.Tensor(c1_h))
    c2_w = data['weights_conv2']
    c2_w = np.transpose(c2_w,axes=[2,0,1])
    c2_w = np.expand_dims(c2_w,axis=1)
    c2_w = np.reshape(c2_w,newshape=[32, 64, 1, 1])                         # if we need exchange of the last dims latter on
    c2_w = torch.nn.Parameter(torch.Tensor(c2_w))
    c2_h = data['biases_conv2']
    c2_h = c2_h.reshape([-1])                                           # if we need exchange of the last dims latter on
    c2_h = torch.nn.Parameter(torch.Tensor(c2_h))
    c3_w = data['weights_conv3']
    c3_w = np.expand_dims(c3_w,axis=0)
    c3_w = c3_w.reshape([1,32,5,5])
    c3_w = torch.nn.Parameter(torch.Tensor(c3_w))
    c3_h = data['biases_conv3']
    c3_h = c3_h.reshape([-1])                                           # if we need exchange of the last dims latter on
    c3_h = torch.nn.Parameter(torch.Tensor(c3_h))

    #torch.Tensor(data)
    X = torch.Tensor(np.random.rand(200,200))

    conv1 = torch.nn.Conv2d(1, 64, [9, 9], padding=(4,4))
    conv2 = torch.nn.Conv2d(64, 32, [1, 1], padding=0)
    conv3 = torch.nn.Conv2d(32, 1, [5, 5], padding=(2,2))


    # another way to build up the network blocks is in the initial place of Sequence
    net.add_module('conv1',conv1)
    net.add_module('relu1',torch.nn.ReLU())
    net.add_module('conv2',conv2)
    net.add_module('relu2',torch.nn.ReLU())
    net.add_module('conv3',conv3)
    net.add_module('relu3',torch.nn.ReLU())
    # Assign the weights in the NET perspective instead of Layer's, REMEBER to assign parameters like "c3_h = torch.nn.Parameter(torch.Tensor(c3_h))"
    net.conv1.weight = c1_w
    net.conv1.bias = c1_h
    net.conv2.weight = c2_w
    net.conv2.bias = c2_h
    net.conv3.weight = c3_w
    net.conv3.bias = c3_h

    return net



#device = torch.device('cpu')
device = torch.device('cuda') # Uncomment this to run on GPU
pretrainModelPath = '../Data/pretrain'
pretrainModelFilename = 'x2.mat'
prettrainModelFile = os.path.join(pretrainModelPath,pretrainModelFilename)



## load the pretrained params

X = 1

## build up the computational graph & load the pretrained params
net = SRCNN()

## do testing

img = cv.imread('/home/winston/workSpace/PycharmProjects/SRCNN_TF_REBUILD/Data/dataset/Set5/baby_GT.bmp')
h = img.shape[0]
w = img.shape[1]
imgYUV = cv.cvtColor(img,cv.COLOR_BGR2YCR_CB)
imgY = imgYUV[:,:,0]
imgU = imgYUV[:,:,1]
imgV = imgYUV[:,:,2]


# blur_Y [0,1] -- float32 --- 【1,1,h,w】
blur_Y = imgY.astype(np.float32)
blur_Y /= 255;
blur_Y = scipy.misc.imresize(blur_Y, (int(h / 2), int(w/ 2)), interp='bicubic', mode='F')
blur_Y = scipy.misc.imresize(blur_Y, (h , w), interp='bicubic', mode='F')
blur_Y = np.expand_dims(blur_Y,axis=0)
blur_Y = np.expand_dims(blur_Y,axis=0)
## send the blur_Y into the net and collect the result|| PyTorch use NCHW
blur_Y = torch.Tensor(blur_Y)
resTnsr = net.forward(blur_Y)



pred_Y = np.array(resTnsr.detach()) * 255

pred_Y = pred_Y.astype(dtype=np.uint8)

pred_Y = pred_Y.reshape([h,w,1])




cv.imshow('1',pred_Y);cv.waitKey();cv.destroyAllWindows()


'''
state_dict() and load_state_dict() are the recommended methods to save and load values for some model’s parameters.

'''