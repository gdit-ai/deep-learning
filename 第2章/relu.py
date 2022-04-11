import numpy as np
def relu(x):
    s =np.where(x <0,0, x)
    return s
if __name__ =='__main__':
    x =-1
    s =relu(x)
    print(s)
    x =np.array([2,-3,1])
    s =relu(x)
    print(s)
