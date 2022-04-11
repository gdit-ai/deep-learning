#步骤01 入所需要的模块
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
np.random.seed(10)

#步骤02 下载mnist数据集、读取数据集
(x_Train, y_Train) , (x_Test, y_Test) = mnist.load_data()

#步骤03 将图像特征值转化为6000,28,28,1的4维矩阵
x_Train4D = x_Train.reshape(x_Train.shape[0], 28, 28,1).astype('float32')
x_Test4D = x_Test.reshape(x_Test.shape[0], 28, 28, 1).astype('float32')

#步骤04 进行标准化
#将像素范围设置在【0,1】
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255

#步骤05 label进行一位有效编码转换
#将标签转成读热码
y_TrainHot = np_utils.to_categorical(y_Train)
y_TestHot = np_utils.to_categorical(y_Test)

#=>建立模型
#步骤01 定义模型
model = Sequential()

#步骤02 建立卷积层1
model.add(Conv2D(filters=16, # filter = 16 建立16个滤镜
# kernel_size = (5,5) 每一个滤镜是5 × 5的大小
kernel_size=(5,5), 
    # padding = 'same' 设置卷积运算产生的图像大小不变
padding='same', 
#输入的图像形状为28*28,1代表单色灰度，3代表RGB
input_shape= (28, 28, 1),
# activation设置激活函数为relu建立池化层1
activation='relu'))

#步骤03 建立池化层1
model.add(MaxPooling2D(pool_size=(2,2)))#缩减采样，输出16个14*14图像

#步骤04建立卷积层2
model.add(Conv2D(filters=36,#建立36个滤镜
kernel_size=(5,5),#每一个滤镜是5 × 5的大小
padding='same',#Convolution完成后的图像大小不变
activation='relu'#输出36个14*14的图像
))

#步骤05 建立池化层2，加入Dropout避免Overfitting
model.add(MaxPooling2D(pool_size=(2,2)))#图像大小变为7*7
# 加入DropOut(0.25)，每次训练时，会在神经网络中随机放弃25%的神经元，避免过拟合建立神经网络（平坦层，隐藏层，输出层)建立平坦层
model.add(Dropout(0.25))

#步骤06 建立平坦层
model.add(Flatten())#长度是36*7*7个神经元

#步骤07 建立隐藏层
model.add(Dense(128, activation='relu'))
# 把DropOut加入模型中，DropOut(0.5)在每次迭代时候会随机放弃50%的神经元，避免过拟合
model.add(Dropout(0.5))
# 建立输出层，一共10个单元，对应0-9一共10个数字。使用softmax进行激活
model.add(Dense(10, activation='softmax'))
# 查看模型摘要
print(model.summary())

#=>进行训练
#步骤01 定义训练方式
# 定义训练方式compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#步骤02 开始训练
train_history = model.fit(x = x_Train4D_normalize, y = y_TrainHot,
validation_split=0.2,
#将80%作为训练数据，20%作为测试数据
epochs=10,#执行10个训练周期
batch_size=300,#每一批300项数据
verbose=2#参数为2表示显示训练过程
)
