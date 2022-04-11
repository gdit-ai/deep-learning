#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

video_capture = cv2.VideoCapture("zhongnanshan.mp4")
# video_capture = cv2.VideoCapture(0)
frame_num = 0

input_name_flag = False

name_command = []

while True:
    # 读取摄像头画面
    ret, frame = video_capture.read()
    if ret == False:
        print("播放完成")
        break
    # 改变摄像头图像的大小，图像小，所做的计算就少
    # print(frame.shape)
    # Display
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
        input_name_flag = True
    elif k == -1 or k == 255: #//无输入
        # print(k)
        continue
    #回车打印列表所有字符
    elif k == 13: #回车键完成输入
        print("回车，完成名字输入")
        print(name_command)
        if input_name_flag:
            # print(command)
            input_name = ""
            for v in name_command:
                print(v)
                input_name = input_name + v
            print(input_name)
            s_filename = "./save_img/" + input_name + ".jpg"
            cv2.imwrite(s_filename, frame)
            input_name_flag = False
            name_command.clear()
    else:
        if input_name_flag:
            s_input = chr(k)
            print("输入字符" + s_input)
            name_command.append(s_input)


video_capture.release()
cv2.destroyAllWindows()