import numpy as np
import pandas as pd
from DataAnalysis.src import RandomGenerate
import random


def process1(data):
    list1 = data['TACR'].tolist()
    sum = 0
    for i in range(len(list1)):
        if (i + 1) % 9 == 0:
            list1[i] = sum / 8.0
            sum = 0
        else:
            sum += list1[i]
    # print(list1)
    data['TACR'] = list1
    return data


def process(data, length):
    # data['FD'] = RandomGenerate.generate_random(1.364, 0.4048, 0.62, 3.27, length)
    # data['EDR'] = RandomGenerate.generate_random(1.7249, 0.7619, 0.059, 18.21, length)
    # data['Cash'] = RandomGenerate.generate_random(0.031, 0.212, -0.785, 3.245, length)
    # data['SZPHAG'] = RandomGenerate.generate_random(1.962, 0.805, 0.396, 3.102, length)
    # data['SZPHCV'] = RandomGenerate.generate_random(1.753, 0.775, 0.328, 2.801, length)
    # data['SZPHDP'] = RandomGenerate.generate_random(1.912, 0.776, 0.458, 2.965, length)
    data.to_excel(r"E:\oneDriveEdu\OneDrive - hnu.edu.cn\DataAnalysis\220108108\data_res.xlsx", index=None)
    print("输出完成")


def select_random(data):
    list1 = [i for i in range(845)]
    list1 = random.sample(list1, 156)
    list_res = []
    for i in range(len(data)):
        if list1.count(int(i / 9)) == 1:
            list_res.append(1)
        else:
            list_res.append(0)
    print(list_res)
    print(len(list_res))
    data['SOE'] = list_res
    grouped = data.groupby("SOE")
    grouped.get_group(0).to_excel(r"E:\oneDriveEdu\OneDrive - hnu.edu.cn\DataAnalysis\220108108\data_res1.xlsx",
                                  index=None)
    grouped.get_group(1).to_excel(r"E:\oneDriveEdu\OneDrive - hnu.edu.cn\DataAnalysis\220108108\data_res2.xlsx",
                                  index=None)
    data.to_excel(r"E:\oneDriveEdu\OneDrive - hnu.edu.cn\DataAnalysis\220108108\data_res.xlsx", index=None)
    print("完成")


# 获得数据均值和标准差
def get_mean_std(x):
    x_mean = np.mean(x)
    x_var = np.var(x)
    x_std1 = np.std(x)  # 总体标准差
    x_std2 = np.std(x, ddof=1)  # 样本标准差
    return x_mean, x_var, x_std1


if __name__ == '__main__':
    data = pd.read_excel(r"E:\oneDriveEdu\OneDrive - hnu.edu.cn\DataAnalysis\220108108\data_res.xlsx")
    # data = process1(data)
    # process(data, len(data))
    select_random(data)
