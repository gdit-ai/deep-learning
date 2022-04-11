import numpy as np
def softmax(x):
    x_exp=np.exp(x)
    x_sum=np.sum(x_exp, axis=1,keepdims=True)
    print("x_sum = ",x_sum)
    s =x_exp/x_sum
    return s
if __name__ =='__main__':
    x =np.array([
[9,2,5,0,0],
[7,5,0,0,0]])
    print("softmax(x) = "+str(softmax(x)))
