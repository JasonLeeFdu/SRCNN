import tensorflow as tf
import LjchOpts as ops
import RecordMaker as rec
import os
import time
from datetime import datetime
import numpy as np
'''
v3 基本版本
具有 tf.record queue的基础训练网络
具有输出统计
具有自动降低的学习率
网络保存与断点载入

'''

'''
期望功能:
1.训练100iter统计
2.中途怎么改变lr
3.网络模型的保存 ckpt, 及其载入
4.网络结果测试函数
5.plt 展示训练结果
6.tensorboard
7.去掉绝对路径

'''
# hyper parameter
BATCH_NUM = 1300
ITER_NUM = 20000
LR       = 3e-4

# configuration
DATA_PATH = '/home/winston/PycharmProjects/SRCNN_TF_REBUILD/Data/singlImg/'
RECORD_NAME = 'training.tfrecords'
PRINT_INTERVAL = 20
SAVE_INTERVAL = 20
MODEL_DIR = '/home/winston/PycharmProjects/SRCNN_TF_REBUILD/Data/model/'


def main():
    train()


def test():
    a = 10

def train():
    # 预处理
    if not os.path.exists(RECORD_NAME):
        rec.prepareTrainingData(RECORD_NAME)

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    with tf.Session() as sess:
        ## 初始化文件队列数据借口
        inputTensor, labelTensor = rec.readAndDecode(RECORD_NAME)
        inputBatchTensor, labelBatchTensor = tf.train.shuffle_batch([inputTensor, labelTensor], batch_size=BATCH_NUM,
                                                                    capacity=3000,
                                                                    min_after_dequeue=1000)
        ## 初始化网络及其                                                                                            | 不用写 inputImg = tf.placeholder(tf.float32,shape=[None,None,None,3],name='inputImage') 和 labelImg = tf.placeholder(tf.float32,shape=[None,None,None,3],name='labelImage')
        globalStep = tf.Variable(0, trainable=False)
        pred, loss = ops.network(inputBatchTensor, labelBatchTensor)
        learning_rate = tf.train.exponential_decay(LR, globalStep, ITER_NUM / 20, 0.5, staircase=True)
        trainOpts = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                                   global_step=globalStep)  # tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)

        ## 初始化全局变量,初始化文件队列的读取,设定saver,
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners(sess=sess)
        saver = tf.train.Saver()

        ## 若有断点,需要回复断点
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
        ## 训练运行状态参数
        AvgFreq = 0
        Avgloss = 0

        for iter1 in np.arange(start_it, ITER_NUM + 1):
            startTime = time.time()  # 统计
            _, lossData = sess.run([trainOpts, loss])  # 训练
            endTime = time.time()
            AvgFreq += endTime - startTime
            Avgloss += lossData

            if (iter1 % SAVE_INTERVAL == 0) and (iter1 != 0):  # 保存模型与结构等参数
                saver.save(sess, MODEL_DIR + 'model.ckpt', global_step=globalStep)

            if (iter1 % PRINT_INTERVAL == 0) and (iter1 != 0):  # 显示
                AvgFreq = (PRINT_INTERVAL * BATCH_NUM) / AvgFreq
                Avgloss = Avgloss / PRINT_INTERVAL
                format_str = '%s: Iters %d, average loss = %.7f, average frequency = %.3f(HZ) (batch per sec)'
                print(format_str % (datetime.now(), iter1, Avgloss, AvgFreq))
                AvgFreq = 0
                Avgloss = 0


if __name__ == '__main__':
        main()


# https://blog.csdn.net/index20001/article/details/74322198
# https://blog.csdn.net/ying86615791/article/details/76215363 进阶使用