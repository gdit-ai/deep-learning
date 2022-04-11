#导入相应库：
import numpy as np
#生成一个包含整数0~11的向量
x =np.arange(12)
print(x)
#查看数组大小
print(x.shape)

#将x转换成二维矩阵，其中矩阵的第一个维度为1
x =x.reshape(1,12)
print(x)
