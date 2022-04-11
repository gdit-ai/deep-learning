import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s
if __name__ =='__main__':
    x =1
    x =np.linspace(-10,10)
    y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1,1.0)
plt.show()
