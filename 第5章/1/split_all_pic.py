import numpy as np
import cv2

im_path = "digits.png"
img = cv2.imread(im_path)
print(img.shape)

for num in range(50):
    for col in range(100):
        one_pic = img[num * 20:num * 20 + 20, col * 20:col * 20 + 20,:]
        name = "./pic/" +  str(num * 100 + col) + ".png"
        print(name)
        #count = count + 1
        cv2.imwrite(name, one_pic)

s = 0
with open("dig_label.txt","w") as f:
    for num in range(5000):
        f.write(str(num)+ ".png")
        s = int(num/500)
        print(s)
        f.write(" " + str(s))
        f.write("\n")





