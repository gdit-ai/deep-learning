from PIL import Image
import cv2
import numpy as np

cv_im1 = cv2.imread("./photo/Unknown.jpg")
cv_im2 = cv2.imread("./photo/yangsan.jpg")

size = (160, 160)
cv_im1 = cv2.resize(cv_im1, size, interpolation=cv2.INTER_AREA)

im1 = Image.fromarray(cv2.cvtColor(cv_im1, cv2.COLOR_BGR2RGB))
im2 = Image.fromarray(cv2.cvtColor(cv_im2, cv2.COLOR_BGR2RGB))
#paste函数的参数为(需要修改的图片，粘贴的起始点的横坐标，粘贴的起始点的纵坐标）
im2.paste(im1,(0,0))
#粘贴的图片的左上角和右下角的坐标
# im2.paste(im1,(300,300,800,800))
img= cv2.cvtColor(np.asarray(im2), cv2.COLOR_RGB2BGR)

img = cv2.resize(img, (0, 0), fx=0.33, fy=0.33)
cv2.imshow("OpenCV", img)
cv2.waitKey()
