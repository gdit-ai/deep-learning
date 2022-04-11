import cv2
from keras.models import load_model
size = 64
# img = cv2.imread("./people_data/me/5.jpg")
img = cv2.imread("./people_data/other_faces/5.jpg")
# print(img.shape)
img = cv2.resize(img, (size, size))
shape_img= (img.reshape(1, size, size, 3)).astype('float32')/255

model = load_model('./me.face.model.h5')
prediction = model.predict_classes(shape_img)
print(prediction)

