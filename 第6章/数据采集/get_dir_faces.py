import cv2
import dlib
import os

size = 160
#建立文件夹
output_dir = './others_img_crop'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

 #使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

def one_img_crop(filename, ouput_name):
    img = cv2.imread(filename)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1, x2:y2]
        face = cv2.resize(face, (size, size))
        cv2.imshow('image', face)
        cv2.imwrite(ouput_name, face)
        # index += 1
        cv2.waitKey(100)


input_dir = "./others_img"
num = 0
for (path, dirnames, filenames) in os.walk(input_dir):
    # print(path) #输出对应顶层文件夹
    # print(dirnames)#在当前文件夹下的文件夹
    # print(filenames)#在当前文件夹下的文件夹
    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.bmp'):
            # print(filename)
            full_path = os.path.join(path, filename)
            print(full_path)
            ouput_name = output_dir + '/' + str(num)+ ".jpg"
            one_img_crop(full_path, ouput_name)
            num = num + 1


