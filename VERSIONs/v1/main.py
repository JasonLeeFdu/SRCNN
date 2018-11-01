import tensorflow as tf
import LjchOpts as ops
import RecordMaker as rec
import os
'''
v1 最基本版本
具有 tf.record queue的基础训练网络

'''
# hyper parameter
BATCH_NUM = 1000
ITER_NUM = 20000
LR       = 0.01

# configuration
DATA_PATH = '/home/winston/PycharmProject/SRCNN_TF_REBULD/Data/singlImg/'
RECORD_NAME = 'training.tfrecords'




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
        trainOpts = tf.train.AdamOptimizer(1e-4).minimize(loss,global_step=globalStep)

        # 初始化参数
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners(sess=sess)
        print('sss')
        # 进行训练
        for iter1 in range(ITER_NUM):
            _,lossNum = sess.run([trainOpts,loss])
            print('loss',lossNum)






if __name__ == '__main__':
        main()
