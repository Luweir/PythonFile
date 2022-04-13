import numpy as np
import matplotlib.pyplot as plt

x = [0, 2, 4, 6, 8, 10]
y = [0, 1, 4, 9, 16, 25]
z1 = np.polyfit(x, y, 2)  # 用n次多项式拟合，可改变多项式阶数；
p1 = np.poly1d(z1)  # 得到多项式系数，按照阶数从高到低排列
print(p1)  # 显示多项式
yvals = p1(x)
print(yvals)
xx = [4, 5, 7, 8, 11, 14]
# print(np.array(xx) - 4)
print(p1(np.array(xx) - 4))
