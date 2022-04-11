
#训练参数初始化
w1 =0
w2 =0
w3 =0
b = 0

#预测值
def predict(input_vec):
    x1 = input_vec[0]
    x2 = input_vec[1]
    x3 = input_vec[2]
    n1 = x1 * w1
    n2 = x2 * w2
    n3 = x3 * w3
    y = n1 + n2 + n3
    y = y + b
    return y
#更新权重参数
def update_weights(input_vec, output, label, rate,w1,w2,w3,b):
    x1 = input_vec[0]
    x2 = input_vec[1]
    x3 = input_vec[2]
    delta = label - output
    w1 = w1 + rate * delta * x1
    w2 = w2 + rate * delta * x2
    w3 = w3 + rate * delta * x3
    b+=0
    return w1,w2,w3,b

#训练参数
def train(input_vecs, labels):
    rate = 0.0001
    print(input_vecs)
    #所有样本训练次数
    for epoch in range(10):
        #依次读取每个样本
        for i in range(len(input_vecs)):
            #预测
            output = predict(input_vecs[i])
            # print ("output :",output)
            label = labels[i]
            loss = label - output
            print("loss :", loss)
            global w1,w2,w3,b
            w1,w2,w3,b = update_weights(input_vecs[i], output, label, rate,w1,w2,w3,b)

#获取数据和标签
def get_data(path):
    train_data=[]
    train_labels=[]
    test_labels=[]
    with open(path) as ifile:
        for line in ifile:
            tokens =line.strip().split(' ')
            data =[int(tk) for tk in tokens[:-1]]
            label =tokens[-1]
            train_data.append(data)
            train_labels.append(int(label))
    return train_data,train_labels

data_path='./sanhao_chengji.txt'
input_vecs, labels =get_data(data_path)
print(len(input_vecs))
train(input_vecs, labels)
print("weight : ",w1,w2,w3,b)

input_value=[80,90,70]
print('predict =  %f'%predict(input_value))


