import cv2
import dlib
import os
#建立文件夹
output_dir = './others_img_crop'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
filename = "./others_img/1.jpg"
img = cv2.imread(filename)
print(img.shape)
#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dets = detector(gray_image, 1)
for i, d in enumerate(dets):
    x1 = d.top() if d.top() > 0 else 0
    y1 = d.bottom() if d.bottom() > 0 else 0
    x2 = d.left() if d.left() > 0 else 0
    y2 = d.right() if d.right() > 0 else 0
    face = img[x1:y1, x2:y2]
    cv2.imshow('image', face)
    # cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)
    # index += 1
key = cv2.waitKey(0) & 0xff
