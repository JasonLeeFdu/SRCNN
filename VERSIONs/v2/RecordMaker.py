import os
import cv2 as cv
import numpy as np
import math
import tensorflow as tf
from PIL import Image as image





def prepareTrainingData(recordName):
    # PIL RGB More efficiently
    # img.size[0]-- width  img.size[1]-- height
    # tf.record里面有一个一个的example,每一个example,每一个example都是含有若干个feature的字典
    # opencv 矩阵计算法 批 行 列 通道
    DIR = '/home/winston/PycharmProjects/SRCNN_TF_REBUILD/Data/singlImg/'
    PATCH_SIZE = 32
    SCALE = 2
    writer = tf.python_io.TFRecordWriter(recordName)
    fileList = os.listdir(DIR)
    totalNum = len(fileList)
    counter = 0
    for img_name in fileList:
        #读取, 归一化, 类型|| 以及对训练数据进行任何的操作,操作完毕后写到相应tf record 位置上
        imgGT = cv.imread(DIR+img_name)
        width  = imgGT.shape[1]
        height = imgGT.shape[0]
        nw     = math.floor(width/PATCH_SIZE)
        nh     = math.floor(height/PATCH_SIZE)
        for x in range(nw):
            for y in range(nh):
                subGT = imgGT[y*PATCH_SIZE:(y+1)*PATCH_SIZE,x*PATCH_SIZE:(x+1)*PATCH_SIZE,:]
                subX  = cv.resize(subGT,(int(PATCH_SIZE/SCALE),int(PATCH_SIZE/SCALE)))
                subX  = cv.resize(subX, (int(PATCH_SIZE), int(PATCH_SIZE)))
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
                writer.write(sample.SerializeToString())  # 序列化为字符串
        counter = counter + 1
        if counter%10==0:
            print('当前进度:',round(counter*100/totalNum))
    writer.close()
    print("写入完毕")

def readAndDecode(fileName):
    '''传统的tensorflow文件训练数据读写函数'''
    fileQueue = tf.train.string_input_producer([fileName])      # 文件读取队列,生成tensor
    recordReader = tf.TFRecordReader()                          # 记录读取器
    _,serializedExample = recordReader.read(fileQueue)          # 用记录读取器,读取出一个序列化的示例数据.对训练数据进行解析需要下一步
    features = tf.parse_single_example(                         # 序列化的示例数据(训练数据单元).解析为一个含有很多数据项的feature字典{x1:,x2:,...y1:,y2:...}
        serializedExample,
        features={                                              #   解析目标说明
            'label':tf.FixedLenFeature([],tf.string),
            'input':tf.FixedLenFeature([],tf.string)
        }
    )
    inputImg = tf.decode_raw(features['input'], tf.float32)     # 从生字符串进行解析序列,然后变形图片
    inputImg = tf.reshape(inputImg,[32,32,3])
    labelImg = tf.decode_raw(features['label'],tf.float32)
    labelImg = tf.reshape(labelImg,[32,32,3])
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
        
        

def main():
    reamImgPIL()



if __name__ == '__main__':
    main()


'''