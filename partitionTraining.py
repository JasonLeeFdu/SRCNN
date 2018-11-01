import os
import math
import shutil
import random


PATH_FROM = '/home/winston/PycharmProjects/SRCNN_TF_REBUILD/Data/newSet1/trainingSet/'
PATH_TO   = '/home/winston/PycharmProjects/SRCNN_TF_REBUILD/Data/newSet1/validationSet/'
RATIO = 0.2


fileList = os.listdir(PATH_FROM)
random.shuffle(fileList)
totalLen = len(fileList)
pivot = math.ceil(totalLen * RATIO)
newFileList = fileList[1:pivot+1]
counter = 1;
for item in newFileList:
    shutil.copyfile(PATH_FROM+item, PATH_TO + item)
    if counter%100 == 0 :
        print('当前进度:%.3f' % (counter / newFileList) )
    counter += 1
print('黏贴完毕')