#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from keras.optimizers import SGD
# from keras.utils import np_utils
# from keras.models import load_model
# from keras import backend as K

#建立模型
def build_model(nb_classes = 2):
    #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
    model = Sequential()

    #以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                                 input_shape =  (32, 32, 3)))    #1 2维卷积层
    model.add(Activation('relu'))                                  #2 激活函数层

    model.add(Convolution2D(32, 3, 3))                             #3 2维卷积层
    model.add(Activation('relu'))                                  #4 激活函数层

    model.add(MaxPooling2D(pool_size=(2, 2)))                      #5 池化层
    model.add(Dropout(0.25))                                       #6 Dropout层

    model.add(Convolution2D(64, 3, 3, border_mode='same'))         #7  2维卷积层
    model.add(Activation('relu'))                                  #8  激活函数层

    model.add(Convolution2D(64, 3, 3))                             #9  2维卷积层
    model.add(Activation('relu'))                                  #10 激活函数层

    model.add(MaxPooling2D(pool_size=(2, 2)))                      #11 池化层
    model.add(Dropout(0.25))                                       #12 Dropout层

    model.add(Flatten())                                           #13 Flatten层
    model.add(Dense(512))                                          #14 Dense层,又被称作全连接层
    model.add(Activation('relu'))                                  #15 激活函数层
    model.add(Dropout(0.5))                                        #16 Dropout层
    model.add(Dense(nb_classes))                                   #17 Dense层
    model.add(Activation('softmax'))                               #18 分类层，输出最终结果

    #输出模型概况
    print(model.summary())

    return model

model = build_model()

sgd = SGD(lr=0.01, decay=1e-6,
          momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
model.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['accuracy'])  # 完成实际的模型配置工作

batch_size = 20
nb_epoch = 10

# model.fit(dataset.train_images,
#                dataset.train_labels,
#                batch_size=batch_size,
#                nb_epoch=nb_epoch,
#                validation_data=(dataset.valid_images, dataset.valid_labels),
#                shuffle=True)

