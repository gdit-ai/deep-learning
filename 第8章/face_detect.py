#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import face_recognition

# video_capture = cv2.VideoCapture("zhongnanshan.mp4")
video_capture = cv2.VideoCapture("class.mp4")
# video_capture = cv2.VideoCapture(0)


def detect_face(rgb_small_frame):
    face_locations = face_recognition.face_locations(rgb_small_frame)
    return face_locations

def write_retangle_face(frame, face_locations):
    for (top, right, bottom, left) in face_locations:
        top *= 3
        right *= 3
        bottom *= 3
        left *= 3
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
    return frame

while True:
    # 读取摄像头画面
    ret, frame = video_capture.read()

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    if ret == False:
        print("播放完成")
        break
    # 改变摄像头图像的大小，图像小，所做的计算就少
    print(frame.shape)

    small_frame = cv2.resize(frame, (0, 0), fx=0.33, fy=0.33)
    # opencv的图像是BGR格式的，转成RGB格式
    rgb_small_frame = small_frame[:, :, ::-1]

    #有利于检测的处理速度
    face_locations = detect_face(rgb_small_frame)

    #在原始图上画框注意缩放
    frame = write_retangle_face(frame, face_locations)

    print(small_frame.shape)
    # frame = small_frame

    # Display
    cv2.imshow('Video', frame)
    cv2.waitKey(30)

    # 按Q退出
    k = 0xFF & cv2.waitKey(30)
    if k == ord('q'):
        print("q exit")
        break

video_capture.release()
cv2.destroyAllWindows()