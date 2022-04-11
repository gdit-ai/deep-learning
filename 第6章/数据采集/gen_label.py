import os
input_dir = "../people_data"
label = 0

f = open("dig_label.txt","w")
for (path, dirnames, filenames) in os.walk(input_dir):
    print(path) #输出对应顶层文件夹
    # print(dirnames)#在当前文件夹下的文件夹
    # print(filenames)#在当前文件夹下的文件夹
    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.bmp'):
            # print(filename)
            if (path == "../people_data\hujianhua"):
                label = 0
            elif (path == "../people_data\other_faces"):
                label = 1
            print(label)
            full_path = os.path.join(path, filename)
            f.write(full_path)
            f.write(" ")
            f.write(str(label))
            f.write("\n")
            # print(full_path)