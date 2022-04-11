import numpy as np
import matplotlib.pyplot as plt
def tanh(x):
    s1 =np.exp(x)-np.exp(-x)
    s2 =np.exp(x)+np.exp(-x)
    s = s1 / s2
    return s
if __name__ =='__main__':
    x =1
    x =np.linspace(-1,1)
    y = tanh(x)
plt.plot(x, y)
plt.ylim(-1,1.0)
plt.show()
