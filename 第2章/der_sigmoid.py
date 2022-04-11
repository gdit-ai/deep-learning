import numpy as np
def sigmoid_derivative(x):
    s =1/(1+np.exp(-x))
    ds = s *(1- s)
    return ds
if __name__ =='__main__':
    x =3
    s =sigmoid_derivative(x)
    print(s)
    x = np.array([2,3,4])
    s =sigmoid_derivative(x)
