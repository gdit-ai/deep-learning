#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import random

def get_files(input_dir):
    file_list = []
    for (path, dirnames, filenames) in os.walk(input_dir):
        # print(path) #输出对应顶层文件夹
        # print(dirnames)#在当前文件夹下的文件夹
        # print(filenames)#在当前文件夹下的文件夹
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.bmp'):
                # print(filename)
                full_path = os.path.join(path, filename)
                # print(full_path)
                file_list.append(full_path)
    return file_list

#设置hujianhua文件夹的对应标签为0
def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def read_img_label(file_list, label):
    size = 64
    imgs = []
    labs = []
    #01
    num = 0
    for filename in file_list:
        # print(filename)
        img = cv2.imread(filename)
        # print(img.shape)
        top, bottom, left, right = getPaddingSize(img)
        # 将图片放大， 扩充图片边缘部分
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = cv2.resize(img, (size, size))
        imgs.append(img)
        labs.append(label)
        num = num + 1

    # print(len(imgs))
    # print(len(labs))
    return imgs, labs


def read_dataset():
    input_dir = "./people_data/me"
    # input_dir = r"E:\hujianhua\课程相关\图像识别\代码\识别自己人脸的项目实战\FaceRecognition-tensorflow-master\people_data\hujianhua"

    all_imgs_list = []
    all_label_list = []
    my_file_list = get_files(input_dir)
    # 0->[0,1] 1->[1,0]
    label = 0 #[0, 1]
    my_imgs_list, my_labs_list = read_img_label(my_file_list, label)

    input_dir = "./people_data/other_faces"
    # input_dir = r"E:\hujianhua\课程相关\图像识别\代码\识别自己人脸的项目实战\FaceRecognition-tensorflow-master\people_data\other_faces"
    others_file_list = get_files(input_dir)
    label = 1 #[1, 0] #->0
    others_imgs_list, others_labs_list = read_img_label(others_file_list, label)

    for img in my_imgs_list:
        all_imgs_list.append(img)
    for img in others_imgs_list:
        all_imgs_list.append(img)

    for label in my_labs_list:
        all_label_list.append(label)
    for label in others_labs_list:
        all_label_list.append(label)

    imgs_array = np.array(all_imgs_list)
    # print(imgs_array.shape)

    labs_array = np.array(all_label_list)
    # print(labs_array.shape)

    return imgs_array,labs_array


#加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
def load_data(img_rows = 64, img_cols = 64,
         img_channels = 3, nb_classes = 2):
    #加载数据集到内存
    images, labels = read_dataset()
    print(images.shape)
    print(labels.shape)

    train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.3, random_state = random.randint(0, 100))
    _, test_images, _, test_labels = train_test_split(images, labels, test_size = 0.5, random_state = random.randint(0, 100))

    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
    valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
    input_shape = (img_rows, img_cols, img_channels)

    #输出训练集、验证集、测试集的数量
    print(train_images.shape[0], 'train samples')
    print(valid_images.shape[0], 'valid samples')
    print(test_images.shape[0], 'test samples')

    #我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
    #类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
    train_labels = np_utils.to_categorical(train_labels, nb_classes)
    valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
    test_labels = np_utils.to_categorical(test_labels, nb_classes)
    print(train_labels.shape)
    print(valid_labels.shape)
    print(test_labels.shape)
    #像素数据浮点化以便归一化
    train_images = train_images.astype('float32')
    valid_images = valid_images.astype('float32')
    test_images = test_images.astype('float32')

    #将其归一化,图像的各像素值归一化到0~1区间
    train_images /= 255
    valid_images /= 255
    test_images /= 255

    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels

#建立模型
def build_model(nb_classes = 2):
    #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
    model = Sequential()

    #以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                                 input_shape =  (64, 64, 3)))    #1 2维卷积层
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

train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_data()

batch_size = 20
nb_epoch = 10
train_history = model.fit(train_images,
               train_labels,
               batch_size=batch_size,
               nb_epoch=nb_epoch,
               validation_data=(valid_images, valid_labels),
               shuffle=True)

scores = model.evaluate(test_images, test_labels)
print('accuracy=', scores[1])
prediction = model.predict_classes(test_images)
# print(prediction)
model.save('./me.face.model.h5')