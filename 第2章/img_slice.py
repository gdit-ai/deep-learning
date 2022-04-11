import cv2
import numpy as np

t = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])
print(t)
print(t.shape)
print(t[:,2])

print(t[0:2,2])


img = cv2.imread('lena.jpg')

print(img.shape)

# 取连续的多行
img2 = img[ 100:200,99:203, :]

# cv2.imshow("hello",img)
cv2.imshow("hello",img2)
cv2.waitKey(0)