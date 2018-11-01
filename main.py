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
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
v5 中期版本
具有 tf.record queue的基础训练网络
具有输出统计
具有自动降低的学习率
网络保存与断点载入
数据augmentation 的文件
网络结果测试函数
去掉绝对路径 不是./ 而直接是 Data/...  可用os.getcwd()
更正三通道为Y单通道，另PSNR也是单通道；更正为除255
tensorboard支持
batch size 的理性修改,使得梯度修正回传次数达到相应的次数
'''
'''
注意:
1.PIL的缩小放大,有锯齿现象
2. 效果: opencv > scipy.misc.imresize
3.tf.placeholder_with_default
4.
'''
'''
炼丹总结:
1. 一开始也不必batch_size太小,否则带向一个大凹面,收敛效果后期不好
2. 一开始比较大的batch_size, 之后再升高batch_size,减小lr
3. 但是batchsize 也不能太大
'''

# hyper parameter  41800
BATCH_NUM = 350
LR1       = 1e-3
LR2       = 1e-4
stage1    = 2 * 100000000
stage2    = 3 * 100000000
stage3    = 4 * 100000000
ITER_NUM  = 10 * 100000000


# configuration
TRAIN_DATA_PATH = 'Data/dataset/trainingSet/'
VAL_DATA_PATH = 'Data/dataset/validationSet/'
TEST_DATA_PATH = 'Data/dataset/testingSet/'
TRAIN_RECORD_NAME = 'train.tfrecords'
VAL_RECORD_NAME = 'val.tfrecords'
TEST_RECORD_NAME = 'test.tfrecords'
PRINT_INTERVAL = 50
SUMMARY_INTERVAL = 50
SAVE_INTERVAL = 500
VALIDATE_INTERVAL = 10000
LR_STAIR_WIDTH = 100000000000
LR_STAIR_DECAY = 1
MODEL_DIR = 'Data/model/'
SCALE = 4

def main():
    train()


def loadValData():
    fileList = os.listdir(TEST_DATA_PATH)
    valList  = list()
    for img_name in fileList:
        oriImg = cv.imread(TEST_DATA_PATH + img_name)
        valList.append(oriImg)
    return valList


def test():
    '''测试函数'''
    FLAG_SINGLE_TEST = False
    MODEL_DIR = 'success3/model/'
    MODEL_NAME = 'model.ckpt-47500'
    if FLAG_SINGLE_TEST:
        oriImg = cv.imread('Data/dataset/validationSet/woman_GT.bmp')
        imgHeight = oriImg.shape[0]
        imgWidth  = oriImg.shape[1]
        flag = False
        while imgHeight % SCALE != 0:
            imgHeight -= 1
            flag = True
        while imgWidth % SCALE != 0:
            imgWidth -= 1
            flag = True
        if flag:
            oriImg = oriImg[0:imgHeight,0:imgWidth,:]

        lrImg = cv.resize(oriImg, (int(imgWidth / SCALE), (int(imgHeight / SCALE))), interpolation=cv.INTER_CUBIC)
        lrImg = cv.resize(lrImg, (int(imgWidth), int(imgHeight)), interpolation=cv.INTER_CUBIC)

        oriYUV = cv.cvtColor(oriImg, cv.COLOR_BGR2YCR_CB)
        oriY = oriYUV[:, :, 0]
        lrYUV = cv.cvtColor(lrImg, cv.COLOR_BGR2YCR_CB)
        lrY = lrYUV[:, :, 0]
        # 输入预处理
        lrY = np.reshape(lrY,(1,imgHeight,imgWidth,1)) #reshape 于行列无关
        lrY = lrY.astype(np.float32)
        lrY = lrY/ 255
        sess = tf.Session()
        with sess.as_default():
            # 重新构建网络结构
            inputTnsr = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
            labelTnsr = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
            resTnsr,lossTnsr,_ = ops.network(inputTnsr,labelTnsr)
            # 不必初始化,而是采用之前训练好的参数
            saver = tf.train.Saver()
            saver.restore(sess,MODEL_DIR+MODEL_NAME)
            # 预测
            resImgY = resTnsr.eval(feed_dict={inputTnsr: lrY})
            # 后处理
            resImgY = np.reshape(resImgY,(imgHeight,imgWidth,1))
            lrY = np.reshape(lrY, (imgHeight,imgWidth, 1))
            oriY= np.reshape(oriY, (imgHeight,imgWidth,  1))
            resImgY = resImgY * 255
            resImgY = resImgY.astype(np.uint8)
            lrY = lrY.astype(np.uint8)
            # 计算 PSNR
            psnr = tools.PSNR(lrY,oriY)
            print('BICUBIC PSNR: ',psnr)
            psnr = tools.PSNR(resImgY, oriY)
            print('SRCNN PSNR: ', psnr)
            # 还原图像并展示
            resImgYUV = lrYUV.copy()
            resImgYUV[:,:,0] = np.reshape(resImgY,(imgHeight,imgWidth))
            resImg  = cv.cvtColor(resImgYUV, cv.COLOR_YCR_CB2BGR)
            #resImg = int(resImg * 255)
            cv.imshow('Original Image',oriImg)
            cv.imshow('LR Image', lrImg)
            cv.imshow('SR Image', resImg)
            cv.waitKey(0)
            cv.destroyAllWindows()
        sess.close()
    else:
        fileList = os.listdir(TEST_DATA_PATH)
        avgPSNR = 0.0
        sess = tf.Session()
        counter =0
        with sess.as_default():
            # 重新构建网络结构
            inputTnsr = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
            labelTnsr = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
            resTnsr, lossTnsr,_ = ops.network(inputTnsr, labelTnsr)
            # 不必初始化,而是采用之前训练好的参数
            saver = tf.train.Saver()
            saver.restore(sess, MODEL_DIR + MODEL_NAME)
            for img_name in fileList:
                # 读入并矫正(mod crop)
                oriImg = cv.imread(imgDir+img_name)
                oriYUV = cv.cvtColor(oriImg, cv.COLOR_BGR2YCR_CB)
                imgHeight = oriImg.shape[0]
                imgWidth = oriImg.shape[1]
                flag = False
                while imgHeight % SCALE != 0:
                    imgHeight -= 1
                    flag = True
                while imgWidth % SCALE != 0:
                    imgWidth -= 1
                    flag = True
                if flag:
                    oriYUV = oriYUV[0:imgHeight, 0:imgWidth, :]
                oriY = oriYUV[:, :, 0]

                lrY = cv.resize(oriY, (int(imgWidth / SCALE), (int(imgHeight / SCALE))),interpolation=cv.INTER_CUBIC)   #33.0758

                lrY = cv.resize(lrY, (int(imgWidth), int(imgHeight)), interpolation=cv.INTER_CUBIC)

                #lrY = scipy.misc.imresize(oriY, (int(imgHeight / SCALE), int(imgWidth / SCALE)), interp='bicubic',mode='F')#32.1587

                #lrY = scipy.misc.imresize(lrY, (int(imgHeight), int(imgWidth)), interp='bicubic', mode='F')

                # 输入预处理
                lrY = np.reshape(lrY, (1, imgHeight, imgWidth, 1))  # reshape 于行列无关
                lrY = lrY.astype(np.float32)
                lrY = lrY / 255
                # predict
                resImgY = resTnsr.eval(feed_dict={inputTnsr: lrY})
                # 后处理
                resImgY = np.reshape(resImgY,(imgHeight,imgWidth,1))
                lrY = np.reshape(lrY, ( imgHeight,imgWidth, 1))
                oriY= np.reshape(oriY, ( imgHeight,imgWidth, 1))
                resImgY = resImgY * 255
                resImgY = resImgY.astype(np.uint8)
                lrY = lrY * 255
                lrY = lrY.astype(np.uint8)
                psnr = tools.PSNR(resImgY, oriY)
                avgPSNR += psnr
                counter += 1
                if counter%2==0:
                    print('PSNR 运算已经进行：%.3f%%'%(counter*100/len(fileList)))
            avgPSNR /= len(fileList)
            print('该数据集上PSNR平均值为：%.4f'%avgPSNR)


def train():
    # 预处理
    if not os.path.exists(TRAIN_RECORD_NAME):
        rec.prepareTrainingData(TRAIN_DATA_PATH,TRAIN_RECORD_NAME)

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    valSet = loadValData()

    with tf.Session() as sess:
        ## 初始化文件队列数据借口
        inputTensor, labelTensor = rec.readAndDecode(TRAIN_RECORD_NAME)
        inputBatchTensor, labelBatchTensor = tf.train.shuffle_batch([inputTensor, labelTensor], batch_size=BATCH_NUM,
                                                                    capacity=10000,
                                                                    min_after_dequeue=1500)
        tf.summary.image('inputImg',inputBatchTensor)
        tf.summary.image('labelImg',labelBatchTensor)
        ## 初始化网络及其超参数                                                                                            | 不用写 inputImg = tf.placeholder(tf.float32,shape=[None,None,None,3],name='inputImage') 和 labelImg = tf.placeholder(tf.float32,shape=[None,None,None,3],name='labelImage')
        globalStep = tf.Variable(0, trainable=False)

        inputHolder = tf.placeholder_with_default(inputBatchTensor, shape=[None,None,None,None])
        labelHolder = tf.placeholder_with_default(labelBatchTensor, shape=[None,None,None,None])
        pred, loss = ops.network(inputHolder, labelHolder)

        learning_rate1 = tf.train.exponential_decay(LR1, globalStep, LR_STAIR_WIDTH, LR_STAIR_DECAY,staircase=True)  # learning_rate：0.1；staircase=True;则每100轮训练后要乘以0.96.
        learning_rate2 = tf.train.exponential_decay(LR2, globalStep, LR_STAIR_WIDTH, LR_STAIR_DECAY, staircase=True)  #learning_rate：0.1；staircase=True;则每100轮训练后要乘以0.96.


        trainOpts = tf.train.AdamOptimizer(LR1).minimize(loss,global_step=globalStep)  # tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
        #trainOpts = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=globalStep)  # tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
        '''
        variables_names = [v.name for v in tf.trainable_variables()]
        for k in variables_names:
            print("Variable: ", k) 
        tf.gradients()
        '''

        '''
        ## 根据不同的网络层，调整学习率
        varSet = tf.trainable_variables()                               # 获得所有的训练参数
        varList1 = varSet[0:4]                                          # 正常的学习率的参数
        varList2 = varSet[4:]                                           # 改变的学习率的参数
        opt1 = tf.train.AdamOptimizer(learning_rate1)                   # 正常的学习率
        opt2 = tf.train.AdamOptimizer(learning_rate2)                   # 改变的学习率
        grads = tf.gradients(loss, varList1 + varList2)                 # 计算梯度
        grads1 = grads[:len(varList1)]                                  # 梯度更新表的前半部分
        grads2 = grads[len(varList1):]                                  # 梯度更新表的后半部分
        train_op1 = opt1.apply_gradients(zip(grads1, varList1))         # 利用优化器将梯度更新到变量，优化器1
        train_op2 = opt2.apply_gradients(zip(grads2, varList2))         # 利用优化器将梯度更新到变量，优化器2
        trainOpts = tf.group(train_op1, train_op2)
        increment_op = tf.assign_add(globalStep, tf.constant(1))
                '''

        ## 初始化全局变量,初始化文件队列的读取
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners(sess=sess)
        ## summarize 的聚集
        summWriter = tf.summary.FileWriter('log/train', sess.graph)
        #testWriter = tf.summary.FileWriter('log/test', sess.graph)
        mergedSummOpt = tf.summary.merge_all()


        ## 设定saver,若有断点,需要回复断点
        saver = tf.train.Saver(max_to_keep=100)
        ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
        start_it = 1
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_str = ckpt.model_checkpoint_path
            start_it = int(ckpt_str[ckpt_str.find('-') + 1:]) + 1
            print('Continue training at Iter %d' % start_it)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No training model found, start from iter 1')
        ## 进行训练
        # 训练运行状态参数
        AvgFreq = 0
        Avgloss = 0


        for iter1 in np.arange(start_it, ITER_NUM + 1):
            startTime = time.time()  # 统计
            a,b,summary,_,lossData = sess.run([inputBatchTensor,labelBatchTensor,mergedSummOpt,trainOpts,loss])              #训练，注意sess.run 的第一个参数是一个fetch list
            #summary, _, _, lossData = sess.run([mergedSummOpt, trainOpts, increment_op, loss])  # 训练，注意sess.run 的第一个参数是一个fetch list

            endTime = time.time()
            AvgFreq += endTime - startTime
            Avgloss += lossData
            if (iter1 % SUMMARY_INTERVAL == 0) and (iter1 != 0):                        # 写 sunmmary
                summWriter.add_summary(summary, iter1)

            if (iter1 % PRINT_INTERVAL == 0) and (iter1 != 0):                          # 显示
                AvgFreq = (PRINT_INTERVAL * BATCH_NUM) / AvgFreq
                Avgloss = Avgloss / PRINT_INTERVAL
                format_str = '%s: Iters %d, average loss(255) = %.7f, average frequency = %.3f(HZ) (batch per sec)'
                if iter1 % SAVE_INTERVAL == 0:
                    print(format_str % (datetime.now(), iter1, math.sqrt(Avgloss)*255, AvgFreq),end='')
                else:
                    print(format_str % (datetime.now(), iter1, math.sqrt(Avgloss)*255, AvgFreq))
                AvgFreq = 0
                Avgloss = 0
            if (iter1 % SAVE_INTERVAL == 0) and (iter1 != 0):  # 保存模型与结构等参数 TEST_INTERVAL
                # test()
                saver.save(sess, MODEL_DIR + 'model.ckpt', global_step=iter1)
                print(' ... Iter %d model saved! ' % iter1)

            if (iter1 % VALIDATE_INTERVAL == 0) and (iter1 != 0):                           # 保存模型与结构等参数
                format_str = '%s: Iters %d, VALIDATION PSNR = %.7f || lr1=%.9f,lr2=%.9f'
                avgPSNR = 0
                for imgOri in valSet:
                    imgOriYUV = cv.cvtColor(imgOri, cv.COLOR_BGR2YCR_CB)
                    imgOriY = imgOriYUV[:, :, 0]
                    width = imgOriY.shape[1]
                    height = imgOriY.shape[0]
                    flag = False
                    while height % SCALE != 0:
                        height -= 1
                        flag = True
                    while width % SCALE != 0:
                        width -= 1
                        flag = True
                    if flag:
                        imgOriY = imgOriY[0:height, 0:width]
                    lrY = cv.resize(imgOriY, (int(width / SCALE), int(height / SCALE)), interpolation=cv.INTER_CUBIC)
                    lrY = cv.resize(lrY, (int(width), int(height)), interpolation=cv.INTER_CUBIC)
                    # 预处理
                    lrY = lrY.astype(np.float32)
                    lrY = lrY/255
                    lrY = np.reshape(lrY, (1,height, width, 1))
                    resImgY = pred.eval(feed_dict={inputHolder: lrY})
                    # 后处理
                    resImgY = np.reshape(resImgY, (height, width, 1))
                    resImgY = resImgY * 255
                    resImgY = resImgY.astype(np.uint8)
                    imgOriY = np.reshape(imgOriY, (height, width, 1))
                    psnr = tools.PSNR(resImgY, imgOriY)
                    avgPSNR += psnr
                
                

                avgPSNR /= len(valSet)
                lr1 = sess.run([learning_rate1])
                lr2 = sess.run([learning_rate2])
                print(format_str % (datetime.now(), sess.run(globalStep), avgPSNR, lr1[0], lr2[0]))




if __name__ == '__main__':
        main()




#  TF:             https://github.com/tensorflow/tensorboard/blob/master/README.md
# https://blog.csdn.net/index20001/article/details/74322198
# https://blog.csdn.net/ying86615791/article/details/76215363 进阶使用
# https://blog.csdn.net/daniaokuye/article/details/78699138
# https://breakthrough.github.io/Installing-OpenCV/#compiling-and-installing-opencv 构建bicubic opencv, 修改A


'''

 varList1 = [varSet[0],varSet[2]]
        varList2 = [varSet[1],varSet[3]]
        varList3 = [varSet[4]]
        varList4 = [varSet[5]]


        opt1 = tf.train.GradientDescentOptimizer(5e-3)                          # 正常的学习率
        opt2 = tf.train.GradientDescentOptimizer(1e-5)                          # 正常的学习率
        opt3 = tf.train.GradientDescentOptimizer(1e-3)                          # 正常的学习率
        opt4 = tf.train.GradientDescentOptimizer(1e-7)                          # 改变的学习率
        grads = tf.gradients(loss, varSet)                                      # 计算梯度
        grads1 = [grads[0],grads[2]]                                            # 梯度更新表的前半部分
        grads2 = [grads[1],grads[3]]                                            # 梯度更新表的前半部分
        grads3 = [grads[4]]                                                     # 梯度更新表的前半部分
        grads4 = [grads[5]]                                                     # 梯度更新表的后半部分
        train_op1 = opt1.apply_gradients(zip(grads1, varList1))    # 利用优化器将梯度更新到变量，优化器1
        train_op2 = opt2.apply_gradients(zip(grads2, varList2))    # 利用优化器将梯度更新到变量，优化器2
        train_op3 = opt3.apply_gradients(zip(grads3, varList3))    # 利用优化器将梯度更新到变量，优化器1
        train_op4 = opt4.apply_gradients(zip(grads4, varList4))    # 利用优化器将梯度更新到变量，优化器2
        trainOpts = tf.group(train_op1, train_op2, train_op3, train_op4)        # 总优化

https://github.com/chrisranderson/beholder

'''