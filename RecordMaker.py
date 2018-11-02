import os
import cv2 as cv
import numpy as np
import math
import tensorflow as tf
from PIL import Image as image
import scipy


def prepareTrainingData(directory,recordName):
    # PIL RGB More efficiently
    # img.size[0]-- width  img.size[1]-- height
    # tf.record里面有一个一个的example,每一个example,每一个example都是含有若干个feature的字典
    # opencv 矩阵计算法 批 行 列 通道
    DIR = os.getcwd() + '/' + directory
    PATCH_SIZE = 32
    STRIDE = 13
    SCALE = 2
    AUG = True

    writer = tf.python_io.TFRecordWriter(recordName)
    fileList = os.listdir(DIR)
    totalNum = len(fileList)
    counter = 0
    for img_name in fileList:
        #读取, 归一化, 类型|| 以及对训练数据进行任何的操作,操作完毕后写到相应tf record 位置上
        imgGT = cv.imread(DIR+img_name)
        #RGB -> YCbCr
        imgGT = cv.cvtColor(imgGT, cv.COLOR_BGR2YCR_CB)
        imgGT = imgGT[:,:,0]
        width  = imgGT.shape[1]
        height = imgGT.shape[0]
        for x in np.arange(0,width-PATCH_SIZE+1,STRIDE):
            for y in np.arange(0,height-PATCH_SIZE+1,STRIDE):
                subGT_ = imgGT[y:y+PATCH_SIZE,x:x+PATCH_SIZE]
                subX_  = scipy.misc.imresize(subGT_, (int(PATCH_SIZE/SCALE),int(PATCH_SIZE/SCALE)), interp='bicubic', mode='F')
                subX_  = scipy.misc.imresize(subX_, (PATCH_SIZE, PATCH_SIZE), interp='bicubic', mode='F')
                # 是否进行数据 augmentation
                if AUG:
                    pairList = dataAug(subGT_, subX_)
                    for subGT, subX in pairList:                     # 居然写反了，怪不得越训越差
                        subX  = subX.astype(np.float32)
                        subGT = subGT.astype(np.float32)
                        subGT = subGT / 255
                        subX  = subX  / 255
                        subGT_raw = subGT.tobytes()
                        subX_raw  = subX.tobytes()
                        sample = tf.train.Example(features=tf.train.Features(feature={
                            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[subGT_raw])),
                            'input': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subX_raw]))
                        }))
                else:  #直接保留这个就可以
                    subX = subX_.astype(np.float32)
                    subGT = subGT_.astype(np.float32)
                    subGT = subGT / 255
                    subX = subX / 255
                    subGT_raw = subGT.tobytes()
                    subX_raw = subX.tobytes()
                    sample = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[subGT_raw])),
                        'input': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subX_raw]))
                    }))
                writer.write(sample.SerializeToString())  # 序列化为字符串
        counter = counter + 1
        if counter%10==0:
            print('制作数据集,当前进度:%.3f%%'%(counter*100/totalNum))
    writer.close()
    print("写入完毕")



def dataAug(im1,im2):
    ''' 输入一对数据,返回增强的数据队列'''
    list1 = list()
    list1.append([im1,im2])
    im1R = np.rot90(im1)
    im2R = np.rot90(im2)
    list1.append([im1R,im2R])
    im1RR = np.rot90(im1R)
    im2RR = np.rot90(im2R)
    list1.append([im1RR, im2RR])
    im1RRR = np.rot90(im1RR)
    im2RRR = np.rot90(im2RR)
    list1.append([im1RRR, im2RRR])
    imm1 = cv.flip(im1,1)
    imm2 = cv.flip(im2,1)
    list1.append([imm1, imm2])
    imm1R = np.rot90(imm1)
    imm2R = np.rot90(imm2)
    list1.append([imm1R, imm2R])
    imm1RR = np.rot90(imm1R)
    imm2RR = np.rot90(imm2R)
    list1.append([imm1RR, imm2RR])
    imm1RRR = np.rot90(imm1RR)
    imm2RRR = np.rot90(imm2RR)
    list1.append([imm1RRR, imm2RRR])
    return list1




def readAndDecode(fileName):
    '''传统的tensorflow文件训练数据读写函数'''
    fileQueue = tf.train.string_input_producer([fileName])      # 文件读取队列,生成tensor
    recordReader = tf.TFRecordReader()                          # 记录读取器
    _,serializedExample = recordReader.read(fileQueue)          # 用记录读取器,读取出一个序列化的示例数据.对训练数据进行解析需要下一步
    features = tf.parse_single_example(                         # 序列化的示例数据(训练数据单元).解析为一个含有很多数据项的feature字典{x1:,x2:,...y1:,y2:...}
        serializedExample,
        features={                                              #   解析目标说明
            'label':tf.FixedLenFeature([],tf.string),           ###$$$  千万注意要要对等tf.string -- tobytes  tostring
            'input':tf.FixedLenFeature([],tf.string)
        }
    )
<<<<<<< HEAD


    labelImg = tf.decode_raw(features['label'],tf.float32)
=======
    labelImg = tf.decode_raw(features['label'],tf.float32)       ###$$$  千万注意要要对等tf.string -- tobytes  tostring
>>>>>>> d3b30f2f2936293534b185b6e535aa079c0053f9
    labelImg = tf.reshape(labelImg,[32,32,1])
    inputImg = tf.decode_raw(features['input'], tf.float32)     # 从生字符串进行解析序列,然后变形图片.生成的 tensor 借口
    inputImg = tf.reshape(inputImg,[32,32,1])
    return  inputImg,labelImg








'''
实验用函数：

def readImgOpencv():
    #opencv->img  numpy ndarray || BGR
    DIR = '/home/winston/PycharmProjects/SRCNN_TF_REBUILD/Data/singlImg/'
    fileName = 'ILSVRC2013_val_00004178.JPEG'
    img = cv.imread(DIR+fileName)
    cv.imshow('hahhah',img)
    cv.waitKey(0)

def reamImgPIL():
    # PIL RGB More efficiently
    DIR = '/home/winston/PycharmProjects/SRCNN_TF_REBUILD/Data/singlImg/'
    fileName = 'ILSVRC2013_val_00004178.JPEG'
    img = image.open(DIR+fileName)


    z = np.array(img)
    zz = z

def printNames():
    DIR = '/home/winston/PycharmProjects/SRCNN_TF_REBUILD/Data/singlImg/'
    for img_name in os.listdir(DIR):
        print(img_name)
        
        



'''