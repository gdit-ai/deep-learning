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
print(labels.shape)
labels = labels.reshape(-1, 1) #-1为指定的值将被推断出为1000
print(data.shape)
print(labels.shape)