import os

input_path = "./imgs"

#root 所指的是当前正在遍历的这个文件夹的本身的地址
#dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
#files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
for root, dirs, files in os.walk(input_path, topdown=False):
    # print(root)
    for name in files:
        # print(name)
        print(os.path.join(root, name))
    # for name in dirs:
    #     print(os.path.join(root, name))