import numpy as np
t1 = np.array([[1, 2, 3, 4, 5],
               [6, 7, 8, 9, 10]])
print(t1.shape)
t2 = np.array([[11, 12, 13, 14, 15],
               [16, 17, 18, 19, 20]])
print(t2.shape)
t4 = np.array([[1, 2, 3],
               [4, 5, 6]])
print(t4.shape)
# 水平拼接
t3 = np.hstack((t1, t2))
# [[ 1  2  3  4  5 11 12 13 14 15]
#  [ 6  7  8  9 10 16 17 18 19 20]]
print(t3)
print(t3.shape)

# 水平拼接
t5 = np.hstack((t1, t4))
print(t5)
print(t5.shape)

# 竖直拼接
t6 = np.vstack((t1, t2))
print(t6)