import cv2
import numpy as np

# read_path = "./static/images/" + "2.jpg"
# change_img = cv2.imread(read_path)

change_img = np.zeros((100, 100, 3), np.uint8)
# 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
change_img = cv2.putText(change_img, '4', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# change_img = cv2.putText(change_img, str(label), (0,0), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
# cv2.imshow("hello", change_img)
cv2.imwrite("./static/images/88.jpg", change_img)