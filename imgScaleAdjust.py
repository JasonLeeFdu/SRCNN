import tensorflow as tf
import LjchOpts as ops
import RecordMaker as rec
import os
import time
from datetime import datetime
import numpy as np
import math
import scipy
import cv2 as cv
import tools
import shutil



os.environ["CUDA_VISIBLE_DEVICES"] = "0"



scrDir = '/home/winston/workSpace/PycharmProjects/SRCNN_TF_REBUILD/Data/dataset/trainingSet/'
dstDirSuffix = '_ADJUST'

fileNameList = os.listdir(scrDir)
if scrDir[-1] == '/':
    scrDir = scrDir[:-1]
dstDir = scrDir + dstDirSuffix

if not os.path.exists(dstDir):
    os.mkdir(dstDir)

for fileName in fileNameList:
    scrFilePath = os.path.join(scrDir,fileName)
    dstFilePath  = os.path.join(dstDir,fileName)
    im = cv.imread(scrFilePath)
    h = im.shape[0]
    w = im.shape[1]
    flag1 = h > 520 or w > 520
    flag2 = h*w > 500*450
    flag = flag1 or flag2
    if flag:
        # we should get it resized
        r = 520 / max(h,w)
        im1 = scipy.misc.imresize(im, (int(h*r), int(w*r)), interp='bicubic', mode='RGB')
        cv.imwrite(dstFilePath,im1)
        print('{}------({}X{})RESIZE({}X{})------>{}'.format(scrFilePath, h,w , int(h*r), int(w*r),dstFilePath))
    else:
        # copy it directly
        shutil.copyfile(scrFilePath, dstFilePath)
        print('{}===============>{}'.format(scrFilePath, dstFilePath))






