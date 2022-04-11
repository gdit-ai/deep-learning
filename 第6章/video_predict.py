import cv2
import dlib
from keras.models import load_model
import sys

size = 64

# 使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0)

model = load_model('./me.face.model.h5')

while True:
    _, img = cam.read()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1, x2:y2]
        # 调整图片的尺寸
        face = cv2.resize(face, (size, size))
        shape_img = (face.reshape(1, size, size, 3)).astype('float32') / 255

        prediction = model.predict_classes(shape_img)
        print(prediction[0])
        name = "unknown"
        if prediction[0] == 0:
            print("识别出本人")
            name = "hujianhua"
        else:
            print("不是本人")
            name = "unknown"
        cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, name, (x2, x1), font, 0.8, (255, 255, 255), 1)

    cv2.imshow('image', img)
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        sys.exit(0)