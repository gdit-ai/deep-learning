#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

video_capture = cv2.VideoCapture("zhongnanshan.mp4")
# video_capture = cv2.VideoCapture(0)
frame_num = 0

while True:
    # 读取摄像头画面
    ret, frame = video_capture.read()
    if ret == False:
        print("播放完成")
        break
    # 改变摄像头图像的大小，图像小，所做的计算就少
    # print(frame.shape)
    # Display
    #保存图片功能
    s_filename = "./save_img/" + str(frame_num) + ".jpg"
    frame_num = frame_num + 1


    cv2.imshow('Video', frame)
    cv2.waitKey(30)

    # 按Q退出
    k = 0xFF & cv2.waitKey(30)

    # print(k)
    if k == ord('q'):
        print("q exit")
        break
    elif k == ord('s'):
        print("按下S键")
        print(s_filename)
        cv2.imwrite(s_filename, frame)

video_capture.release()
cv2.destroyAllWindows()