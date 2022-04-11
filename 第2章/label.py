#代码路径:/第2章/label.py
#导入相应库：
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.font_manager import  FontProperties
font=FontProperties(fname=r"./simsun.ttc",size=10)
#创建x轴数组
x = np.arange(10)
#创建y轴数组
y= np.exp(-x) * np.sin(x)
#添加图例说明
plt.plot(x, y, 'bo--', label='e^(-x) * sin(x)')
plt.legend()
#设置横轴的精度为1，范围为0-9
plt.xticks(np.arange(10))
#加上辅助线
plt.grid(True)
#横轴名称
plt.xlabel("X 轴",fontproperties=font)
#纵轴名称
plt.ylabel("Y 轴",fontproperties=font)
#整个图起名字
plt.title('图像',fontproperties=font)
plt.show()