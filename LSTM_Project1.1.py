'''
添加数据：
1、修改LENGTH_FRAME（所有输入的合帧数）
2、调用readFile函数，填入数据目录和对应的手势长度
3、调用dealData函数，不同的手势长度要分别调用此函数
4、标定标签，根据样本的下标手动计算、更改、添加
'''
'''
测试模型：
调用model_Test函数，输入数据的起始下标和结束下标即可（数据即一开始文件读入的数据(Temp)，以一帧一帧的形式存储），第三个参数Flag
设为0时，表示这是测试时第一次调用该函数，如果想在一个区间调用完后继续调用另一个区间，请将之后的调用函数中的Flag设为1
'''

import numpy
import os
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.utils import np_utils
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #用于消除错误警报

DIMENSION_FEATURE = 128 #输入每一帧的特征维数，这里是128
LENGTH_FRAME = 20000 #所有输入的合帧
TIME_STEPS = 5 #时间步长（即一个样本的长度）
BATCH_SIZE = 1 #向网络中一次丢入的样本的规模
EPOCHS = 100 #训练的次数
EFFECTIVE_LENGTH = 4 #测试时用延迟的方式解决输出错误的问题
CATEGORY_NUMBER = 3 #类别的数量
DELAY = 0
#测试语句中的输出
OUT1 = '前推后拉！！！'
OUT2 = '左右挥手！！！'
OUT3 = '下挥手！！！'
#OUT4 = '下挥手！！！'
#OUT5 = '无输出！！！'

Temp = numpy.zeros((DIMENSION_FEATURE, LENGTH_FRAME), dtype=float)
X = []
X_Test = []
Test_temp = []
Test_X = numpy.zeros((0, DIMENSION_FEATURE * TIME_STEPS), dtype=float)
Test_y_temp = numpy.zeros(TIME_STEPS)
Samples_Number = 0
Samples_Number_Test = 0
Temp_Col = 0
Length_R = 0
Sum_all = 0
Sum_right = 0

#通过文件读入数据，第一个参数是文件夹的路径，第二个参数是手势的长度
def readFile(filepath, Length_Hand, flag):
    global Temp_Col
    global Length_R
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        Length_R = Length_R + Length_Hand
        filepath_child = os.path.join('%s%s' % (filepath, allDir))
        #print(Length_R)
        #print(filepath_child)
        file = open(filepath_child, 'r')
        lines = file.readlines()
        Temp_row = 0  # 行标记
        for line in lines:
            List = line.strip('\n').split(' ')
            List_tran = []
            for i in range(len(List)):
                List_tran.append(List[i])
                if List[0] in List_tran:
                    List_tran.pop()
            for i in range(Length_Hand):
                List_tran[i] = float(List_tran[i])
            Temp[Temp_row][Temp_Col:Temp_Col + Length_Hand] = List_tran[0:Length_Hand]
            Temp_row = Temp_row + 1
        Temp_Col = Temp_Col + Length_Hand
        dealDate(Length_R - Length_Hand, Length_R, Length_Hand, flag)
        file.close()

#将读入的数据处理成N*1*128的形式，第一个参数是读入数据的开始帧，第二个参数是读入数据的结束帧，第三个参数是这部分数据中手势的长度
def dealDate(start, end, Length_Hand, flag):
    global Samples_Number
    global Samples_Number_Test
    if flag == 1:
        for i in range(start, end):
            if ((i - start) % Length_Hand == 0):
                for j in range(i, i + Length_Hand - TIME_STEPS + 1):
                    for k in range(j, j + TIME_STEPS):
                        for n in range(DIMENSION_FEATURE):
                            X.append(Temp[n, k])
                Samples_Number = Samples_Number + Length_Hand - TIME_STEPS + 1
    else:
        for i in range(start, end):
            if ((i - start) % Length_Hand == 0):
                for j in range(i, i + Length_Hand - TIME_STEPS + 1):
                    for k in range(j, j + TIME_STEPS):
                        for n in range(DIMENSION_FEATURE):
                            X_Test.append(Temp[n, k])
                Samples_Number_Test = Samples_Number_Test + Length_Hand - TIME_STEPS + 1

#用于测试模型
'''
def model_Test(start, end, Flag):
    for i in range(start, end):
        if (((i - start) < (TIME_STEPS-1)) and (Flag == 0)):
            for j in range(DIMENSION_FEATURE):
                Test_temp.append(Temp[j, i])
        else:
            for j in range(DIMENSION_FEATURE):
                Test_temp.append(Temp[j, i])
            Test_X = Test_temp
            Test_X = numpy.reshape(Test_X, (1, TIME_STEPS, DIMENSION_FEATURE))
            prediction = model.predict(Test_X, verbose=0)
            result = numpy.argmax(prediction)
            print(i)
            for j in range(DIMENSION_FEATURE):
                Test_temp.pop(0)
            Test_y_temp[result] = Test_y_temp[result] + 1
            flag = -1
            for j in range(CATEGORY_NUMBER):
                if Test_y_temp[j] == EFFECTIVE_LENGTH:
                    for k in range(CATEGORY_NUMBER):
                        if k != j:
                            Test_y_temp[k] = 0
                        else:
                            Test_y_temp[k] = Test_y_temp[k] - 1
                    flag = j
            if ((flag == -1) or (flag == EFFECTIVE_LENGTH)):
                print(OUT5)
                for j in range(CATEGORY_NUMBER):
                    if j != result:
                        Test_y_temp[j] = 0
            if flag == 0:
                print(OUT1)
            if flag == 1:
                print(OUT2)
            if flag == 2:
                print(OUT3)
            if flag == 3:
                print(OUT4)
'''

def model_Test_Original(start, end, Flag, answ):
    global Sum_all,Sum_right
    Sum_all = Sum_all + end - start
    for i in range(start, end):
        if (((i - start) < (TIME_STEPS-1)) and (Flag == 0)):
            for j in range(DIMENSION_FEATURE):
                Test_temp.append(Temp[j, i])
        else:
            for j in range(DIMENSION_FEATURE):
                Test_temp.append(Temp[j, i])
            Test_X = Test_temp
            Test_X = numpy.reshape(Test_X, (1, TIME_STEPS, DIMENSION_FEATURE))
            prediction = model.predict(Test_X, verbose=0)
            result = numpy.argmax(prediction)
            print('------------------------')
            print(i)
            for j in range(DIMENSION_FEATURE):
                Test_temp.pop(0)
            print(result)
            if result == answ:
                Sum_right = Sum_right + 1
            print('------------------------')
    if Flag == -1:
        Sum_all = Sum_all - TIME_STEPS + 1
        print(Sum_right / Sum_all * 100)

#通过文件读入数据
readFile('C:\\Users\\wllxu\\数据与标签\\前推后拉\\速度特征\\21帧\\', 21, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\前推后拉\\速度特征\\22帧\\', 22, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\前推后拉\\速度特征\\23帧\\', 23, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\前推后拉\\速度特征\\24帧\\', 24, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\前推后拉\\速度特征\\25帧\\', 25, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\前推后拉\\速度特征\\26帧\\', 26, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\前推后拉\\速度特征\\27帧\\', 27, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\前推后拉\\速度特征\\28帧\\', 28, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\前推后拉\\速度特征\\29帧\\', 29, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\前推后拉\\速度特征\\30帧\\', 30, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\前推后拉\\速度特征\\31帧\\', 31, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\前推后拉\\速度特征\\32帧\\', 32, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\前推后拉\\速度特征\\35帧\\', 35, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\13帧\\', 13, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\14帧\\', 14, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\15帧\\', 15, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\16帧\\', 16, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\17帧\\', 17, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\18帧\\', 18, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\19帧\\', 19, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\20帧\\', 20, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\21帧\\', 21, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\22帧\\', 22, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\23帧\\', 23, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\24帧\\', 24, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\26帧\\', 26, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\27帧\\', 27, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\左右挥手\\速度特征\\28帧\\', 28, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\下挥手（未分割）\\速度特征120帧\\', 120, 1)
readFile('C:\\Users\\wllxu\\数据与标签\\预测\\前推后拉\\速度特征\\', 120, 0)
readFile('C:\\Users\\wllxu\\数据与标签\\预测\\左右挥手\\速度特征\\', 120, 0)
readFile('C:\\Users\\wllxu\\数据与标签\\预测\\下挥手\\速度特征\\', 120, 0)

X = numpy.reshape(X, (Samples_Number, TIME_STEPS, DIMENSION_FEATURE)) #将输入改成模型可以接受的格式，三维（样本数量，时间步长，特征维数）
X_Test = numpy.reshape(X_Test, (Samples_Number_Test, TIME_STEPS, DIMENSION_FEATURE))
y = numpy.zeros(Samples_Number) #用来存储标签
y_Test = numpy.zeros(Samples_Number_Test)

#标定标签
#0代表前推后拉
#1代表左右挥手
#2代表下挥手
#print(Samples_Number)
#print(Samples_Number_Test)

for i in range(Samples_Number):
    if i < 3198:
        y[i] = 0
    elif i < (3198+2828):
        y[i] = 1
    else:
        y[i] = 2

for i in range(Samples_Number_Test):
    if i < 928:
        y_Test[i] = 0
    elif i < 1508:
        y_Test[i] = 1
    else:
        y_Test[i] = 2

y_one_hot = np_utils.to_categorical(y) #将标签转化为one-hot码
y_one_hot_test = np_utils.to_categorical(y_Test)

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(TIME_STEPS, DIMENSION_FEATURE), return_sequences=True))
model.add(CuDNNLSTM(128, return_sequences=False))
#model.add(CuDNNLSTM(128, return_sequences=False))
model.add(Dense(y_one_hot.shape[1], activation='softmax')) #softmax 输出层激励函数
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #categorical_crossentropy 交叉熵 损失函数 adam优化器

tb = TensorBoard(log_dir="E:\logs",
                 histogram_freq=5,
                 batch_size=1,
                 write_graph=True,
                 write_grads=True,
                 write_images=True,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)
callbacks = [tb]

model.fit(X, y_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_Test, y_one_hot_test), callbacks=callbacks)
scores = model.evaluate(X, y_one_hot, verbose=0)
print("Model Accuracy:%.2f%%" % (scores[1]*100))

model.save('C:\\Users\\wllxu\\model\\Lstm_Project_19_10_22.h5')

#model = load_model('C:\\Users\\wllxu\\model\\Lstm_ProjectNew_10.h5')

#model_Test_Original(1131, 1280, 0)
'''
model_Test_Original(9714, 10674, 0, 0)
model_Test_Original(10674, 11274, 1, 1)
model_Test_Original(11274, 11634, -1, 2)
'''
