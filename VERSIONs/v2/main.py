import tensorflow as tf
import LjchOpts as ops
import RecordMaker as rec
import os
import time
from datetime import datetime
'''
v2 基本版本
具有 tf.record queue的基础训练网络
具有输出统计
具有自动降低的学习率

'''

'''
期望功能:
1.训练100iter统计
2.中途怎么改变lr
3.网络模型的保存 ckpt, 及其载入
4.网络结果测试函数
5.plt 展示训练结果
6.tensorboard

'''
# hyper parameter
BATCH_NUM = 900
ITER_NUM = 20000
LR       = 3e-4

# configuration
DATA_PATH = '/home/winston/PycharmProject/SRCNN_TF_REBULD/Data/singlImg/'
RECORD_NAME = 'training.tfrecords'
PRINT_INTERVAL = 20


def main():
    # 准备数据
    if not os.path.exists(RECORD_NAME):
        rec.prepareTrainingData(RECORD_NAME)
    with tf.Session() as sess:
        # 把数据接口对准网络
        inputTensor, labelTensor = rec.readAndDecode(RECORD_NAME)
        inputBatchTensor, labelBatchTensor = tf.train.shuffle_batch([inputTensor, labelTensor], batch_size=BATCH_NUM, capacity=3000,
                                                        min_after_dequeue=1000)
        # 设定网络,构建训练参数                                                                                            | 不用写 inputImg = tf.placeholder(tf.float32,shape=[None,None,None,3],name='inputImage') 和 labelImg = tf.placeholder(tf.float32,shape=[None,None,None,3],name='labelImage')
        globalStep = tf.Variable(0,trainable=False)
        pred,loss = ops.network(inputBatchTensor,labelBatchTensor)
        learning_rate = tf.train.exponential_decay(LR, globalStep, ITER_NUM/20, 0.3, staircase=False)
        trainOpts = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=globalStep)                         # tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)

        # 初始化参数
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners(sess=sess)


        # 进行训练
        # 训练运行状态参数
        AvgFreq = 0
        Avgloss = 0

        for iter1 in range(ITER_NUM):
            startTime = time.time()                                 # 统计
            _,lossData = sess.run([trainOpts,loss])                 # 训练
            endTime   = time.time()
            AvgFreq  += endTime - startTime
            Avgloss  += lossData
            if (iter1 % PRINT_INTERVAL == 0) and (iter1 != 0):      # 显示
                AvgFreq = (PRINT_INTERVAL*BATCH_NUM) / AvgFreq
                Avgloss = Avgloss / PRINT_INTERVAL
                format_str = '%s: step %d, average loss = %.7f, average frequency = %.3fHZ (batch per sec)'
                print (format_str % (datetime.now(), iter1, Avgloss,AvgFreq))
                AvgFreq = 0
                Avgloss = 0



if __name__ == '__main__':
        main()


#https://blog.csdn.net/index20001/article/details/74322198