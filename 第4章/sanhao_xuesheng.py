#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#获取数据和标签
def get_data(path):
    train_data = []
    train_labels = []
    test_labels = []
    with open(path) as ifile:
        for line in ifile:
            tokens = line.strip().split(' ')
            data = [int(tk) for tk in tokens[:-1]]
            label = tokens[-1]
            # print(data)
            # print(label)
            train_data.append(data)
            train_labels.append(label)
    return train_data, train_labels

data_path = './chengji.txt'
train_data, train_labels = get_data(data_path)
data = np.array(train_data).astype('float32')
labels = np.array(train_labels).astype('float32')
data = data / 100
labels = labels.reshape(-1, 1) #-1为指定的值将被推断出为1000
print(data.shape)
print(labels.shape)


# 对于一个单输入模型的二分类问题
model = Sequential()
model.add(Dense(3, activation='relu', input_dim=3))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 训练模型, 以每批次1样本迭代数据
model.fit(data, labels, epochs=1, batch_size=1)


scores = model.evaluate(data, labels)
print('accuracy=', scores[1])
prediction = model.predict_classes(data)
# print(prediction)




