#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

video_capture = cv2.VideoCapture("class.mp4")
# video_capture = cv2.VideoCapture(0)

while True:
    # 读取摄像头画面
    ret, frame = video_capture.read()

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    if ret == False:
        print("播放完成")
        break
    # 改变摄像头图像的大小，图像小，所做的计算就少
    print(frame.shape)
    # Display
    cv2.imshow('Video', frame)
    cv2.waitKey(30)

    # 按Q退出
    k = 0xFF & cv2.waitKey(300)
    if k == ord('q'):
        print("q exit")
        break

video_capture.release()
cv2.destroyAllWindows()