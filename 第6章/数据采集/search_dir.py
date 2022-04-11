import os
input_dir = "./others_img"

def get_files(path):
    file_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            # print(os.path.join(root, name))
            file_list.append(os.path.join(root, name))
    return  file_list

for (path, dirnames, filenames) in os.walk(input_dir):
    # print(path) #输出对应顶层文件夹
    # print(dirnames)#在当前文件夹下的文件夹
    # print(filenames)#在当前文件夹下的文件夹
    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.bmp'):
            # print(filename)
            full_path = os.path.join(path, filename)
            print(full_path)
