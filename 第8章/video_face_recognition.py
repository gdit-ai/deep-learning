#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import os
import face_recognition

known_face_names = []
known_face_encodings = []

def get_people_info(known_face_names, known_face_encodings):
    filepath = './photo'
    filename_list = os.listdir(filepath)
    num_people = 0
    print("get_people_info")
    for filename in filename_list:  # 依次读入列表中的内容
        if filename.endswith('jpg'):  # 后缀名'jpg'匹对
            num_people += 1
            known_face_names.append(filename[:-4])  # 把文件名字的后四位.jpg去掉获取人名
            file_str = '.\\photo\\' + filename
            a_images = face_recognition.load_image_file(file_str)
            print(file_str)
            a_face_encoding = face_recognition.face_encodings(a_images)[0]
            known_face_encodings.append(a_face_encoding)
    print(known_face_names, num_people)
    return known_face_names, known_face_encodings

known_face_names, known_face_encodings = get_people_info(known_face_names, known_face_encodings)


def face_match(rgb_small_frame, known_face_names, known_face_encodings):
    # 根据encoding来判断是不是同一个人，是就输出true，不是为flase
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    face_flag = 0
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.42)
        # 阈值太低容易造成无法成功识别人脸，太高容易造成人脸识别混淆 默认阈值tolerance为0.6
        name = "Unknown"
        if True in matches:  # 在数据库找到匹配人员
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        face_names.append(name)
        face_flag = 1
    return face_flag, face_names, face_locations

# video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture("zhongnanshan.mp4")

while True:
    # 读取摄像头画面
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.33, fy=0.33)
    print("进行识别")
    # opencv的图像是BGR格式的，转成RGB格式
    rgb_small_frame = small_frame[:, :, ::-1]
    face_flag, face_names, face_locations = face_match(rgb_small_frame, known_face_names,
                                                                 known_face_encodings)
    if face_flag:
        last_name = face_names[0]
        set_name = set(face_names)
        set_names = tuple(set_name)
        print("识别出人脸是 = " + str(set_names))
    if ret == False:
        print("播放完成")
        break
    # 改变摄像头图像的大小，图像小，所做的计算就少
    print(frame.shape)
    # Display
    cv2.imshow('Video', frame)
    cv2.waitKey(30)

    # 按Q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()