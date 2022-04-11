


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

#训练参数初始化
w1 =0.72
w2 =0.20
w3 =0.08
b = 0
print("weight : ",w1,w2,w3,b)

input_value=[80,90,70]
print('predict =  %f'%predict(input_value))
