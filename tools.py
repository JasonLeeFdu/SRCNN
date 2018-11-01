import tensorflow as tf
import LjchOpts as ops
import RecordMaker as rec
import os
import time
from datetime import datetime
import numpy as np
import math
import cv2 as cv



def PSNR(im1,im2):
    '''function psnr=compute_psnr(im1,im2)
if size(im1, 3) == 3,
    im1 = rgb2ycbcr(im1);
    im1 = im1(:, :, 1);
end

if size(im2, 3) == 3,
    im2 = rgb2ycbcr(im2);
    im2 = im2(:, :, 1);
end

imdff = double(im1) - double(im2);
imdff = imdff(:);

rmse = sqrt(mean(imdff.^2));
psnr = 20*log10(255/rmse);'''
    if len(im1.shape) != len(im2.shape):
        print('Unmatched shape!')
        return
    if len(im1.shape) == 3 and im1.shape[2] == 3:
        im1yuv = cv.cvtColor(im1, cv.COLOR_BGR2YCR_CB)
        im1 = im1yuv[:,:,0]
    if len(im2.shape) == 3 and im2.shape[2] == 3:
        im2yuv = cv.cvtColor(im2, cv.COLOR_BGR2YCR_CB)
        im2 = im2yuv[:,:,0]


    imdelta = (np.float64(im1) - np.float64(im2))
    rmse = math.sqrt(np.mean(imdelta**2))
    psnr = 20*np.log10(255/rmse)
    return psnr
