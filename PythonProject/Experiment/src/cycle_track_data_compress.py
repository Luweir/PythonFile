import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser
import os
import datetime
from Experiment.data_process.data_process import load_data, get_points


# 获得轨迹数据集 points 等频化，频率为frequency
def get_new_points(points, frequency):
    # 等频化后的轨迹数据
    new_points = [points[0]]
    next_time = frequency
    for i in range(1, len(points)):
        # 如果时间超过 next_time 说明该记录的点在里面
        if points[i][0] >= next_time:
            while points[i][0] >= next_time:
                new_points.append(get_value(points[i - 1], points[i], next_time))
                next_time += frequency
    return new_points


#   根据上一个点和当前点以及中间点的时间值，得到两轨迹点之间的等频轨迹点
def get_value(pre, cur, x):
    k1 = (cur[1] - pre[1]) / (cur[0] - pre[0])
    k2 = (cur[2] - pre[2]) / (cur[0] - pre[0])
    b1 = cur[1] - k1 * cur[0]
    b2 = cur[2] - k2 * cur[0]
    return [x, k1 * x + b1, k2 * x + b2]


'''
data1.dtypes                         得到 data1各列的数据类型
parser.parse(str(data1.iloc[0,0]))   将其解析为时间对象，便于得到时间差
(time2-time1).seconds                两个时间对象作差，返回差的秒数
'''

columns = ['time', 'latitude', 'longitude']


# 绘图函数
# 单线图轨迹位置，传入DataFrame data
def get_line_chart(data):
    plt.figure(figsize=(12, 8))
    # plt的线图则会按输入顺序来
    plt.plot(data['longitude'], data['latitude'])


# 双线图比较轨迹位置，传入DataFrame data1和data2
def get_two_line_chart(data1, data2):
    data1 = pd.DataFrame(data1, columns=columns)
    data2 = pd.DataFrame(data2, columns=columns)
    plt.figure(figsize=(20, 12))
    plt.plot(data1['longitude'], data1['latitude'], color='red')  # 第一条红色
    plt.plot(data2['longitude'], data2['latitude'], color='blue')  # 第二条蓝色
    plt.show()


# 双线图：时间X纬度
def get_time_latitude(data1, data2):
    data1 = pd.DataFrame(data1, columns=columns)
    data2 = pd.DataFrame(data2, columns=columns)
    plt.figure(figsize=(20, 12))
    plt.plot(data1['time'], data1['latitude'], color='red')  # 第一条红色
    plt.plot(data2['time'], data2['latitude'], color='blue')  # 第二条蓝色
    plt.show()


# 双线图：时间X精度
def get_time_longitude(data1, data2):
    data1 = pd.DataFrame(data1, columns=columns)
    data2 = pd.DataFrame(data2, columns=columns)
    plt.figure(figsize=(20, 12))
    plt.plot(data1['time'], data1['longitude'], color='red')  # 第一条红色
    plt.plot(data2['time'], data2['longitude'], color='blue')  # 第二条蓝色
    plt.show()


# 按列合并两个数据集
def merge_dataset(points1, points2):
    df1 = pd.DataFrame(points2, columns=[
        'time2', 'latitude2', 'longitude2'])
    # 按列合并
    df2 = pd.DataFrame(points1, columns=['time1', 'latitude1', 'longitude1'])
    save_date = pd.concat([df1, df2], axis=1)
    # save_date.to_excel("new_points.xlsx")
    return save_date


# 参数：file1和file2是文件名，frequency为设置的固定频率，threshold为误差阈值
def same_frequency_compress(file1, file2, frequency, threshold):
    # 1、将两个数据集等频化，得到新的数据集1和数据集2
    # 2、分别计算数据集1的x坐标和数据集2的x坐标之间的间距，得到x间距列表:diff_x；计算数据集1的y坐标和数据集2的y坐标之间的间距，得到y间距列表:diff_y
    # 3、对间距列表进行遍历：begin_x和begin_y设为两个数据集的初始间距，如果 diff_x[i]-begin_x < ε and diff_y[i]-begin_y < ε 则舍弃之间的点（因为可仍为这些点的间距与初始点的间距一样）
    #                                            如果不满足阈值，则需要将第i个点（time,latitude,longitude）这个点存进去
    # 4、用结束标志：(time(最后一个点),x(-1),y(-1)) 来表示每个周期轨迹的结束
    # 5、当基准周期的点数较少时，后续周期的多出来的点全部存储，再加入结束标志；当基准周期的点数较多时，后续周期点遍历完后直接加入结束标志
    # 6、将一次处理后的总数据集放入压缩算法进行二次压缩，与用原始数据直接应用压缩算法进行压缩进行比较，比较内容包括：
    #       6.1 压缩率
    #       6.2 压缩耗时
    #       6.3 选择轨迹相似度度量方法对轨迹的损失精度进行判定
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # 1、将两个数据集等频化，得到新的数据集1和数据集2
    # frequency = 30

    data1 = load_data(file1)
    points1 = get_points(data1)
    new_points1 = get_new_points(points1, frequency)
    # get_two_line_chart(points1, new_points1)
    print("data1原来轨迹点的个数：" + str(len(points1)))
    print("data1经过等频后轨迹点的个数：" + str(len(new_points1)))

    data2 = load_data(file2)
    points2 = get_points(data2)
    new_points2 = get_new_points(points2, frequency)
    # get_two_line_chart(points2, new_points2)
    print("data2原来轨迹点的个数：" + str(len(points2)))
    print("data2经过等频后轨迹点的个数：" + str(len(new_points2)))
    get_two_line_chart(new_points1, new_points2)
    get_time_latitude(new_points1, new_points2)
    get_time_longitude(new_points1, new_points2)
    # 2、分别计算数据集1的x坐标和数据集2的x坐标之间的间距，得到x间距列表:diff_x；计算数据集1的y坐标和数据集2的y坐标之间的间距，得到y间距列表:diff_y
    diff_x = []
    diff_y = []
    length = min(len(new_points1), len(new_points2))
    for i in range(length):
        diff_x.append(new_points1[i][1] - new_points2[i][1])
        diff_y.append(new_points1[i][2] - new_points2[i][2])

    # 3、对间距列表进行遍历：begin_x和begin_y设为两个数据集的初始间距，如果 diff_x[i]-begin_x < ε and diff_y[i]-begin_y < ε
    # 则舍弃之间的点（因为可仍为这些点的间距与初始点的间距一样） 如果不满足阈值，则需要将第i个点（time,latitude,longitude）这个点存进去
    compress_points = []
    # 设立begin点
    begin_x = diff_x[0]
    begin_y = diff_y[0]
    compress_points.append(new_points2[0])
    i = 1
    # threshold = 0.0005 百米范围内   0.005 数百米范围内
    # threshold = 0.0005
    while i < length:
        # 第i个点的间距与begin点的间距差值在阈值内，可以丢弃该点
        if abs(diff_x[i] - begin_x) <= threshold and abs(diff_y[i] - begin_y) <= threshold:
            i += 1
            continue
        # 否则就该留下该点
        else:
            compress_points.append(new_points2[i])
            begin_x = diff_x[i]
            begin_y = diff_y[i]
            i += 1
    while i < len(new_points2):
        compress_points.append(new_points2[i])
        i += 1
    # 比较结果
    print("误差损失 ：" + str(threshold))
    print("参考集 new_points1 length:" + str(len(new_points1)))
    print("待压缩集 new_points2 length:" + str(len(new_points2)))
    print("压缩后 new_point2 length:" + str(len(compress_points)))


# 将固频后的轨迹new_points（时间差，维度，经度）按start_time 整合保存到 save_name中
def save_to_excel(start_time, new_points, save_name):
    if type(start_time) == str:
        start_time = parser.parse(start_time)
    points = []
    for ele in new_points:
        points.append([str(start_time + datetime.timedelta(seconds=+ele[0])), ele[1], ele[2]])
    pd.DataFrame(points, columns=['date', 'latitude', 'longitude']).to_excel(save_name, index=None)


# if __name__ == '__main__':
#     '''
#     遍历轨迹2的每个点，在轨迹1上搜寻有这个点（接近）的时间（相对值），将轨迹2的点置换为（time，0，0）
#         最大限度地发挥trajic算法的优势
#     '''
#     # 1、得到数据集point1和point2
#     data1 = load_data("10.11.xlsx")
#     points1 = get_points(data1)
#     data2 = load_data("10.10.xlsx")
#     points2 = get_points(data2)
#     # 2、遍历point2，映射到point1的时间
#     # same_frequency_compress("10.13.xlsx", "10.14.xlsx", 30, 0.005)
#     new_points1 = get_new_points(points1, 30)
#     new_points2 = get_new_points(points2, 30)
#
#     save_to_excel(parser.parse(str(data1.iloc[0, 0])), new_points1, "../data/new_10.11.xlsx")
#     save_to_excel(parser.parse(str(data2.iloc[0, 0])), new_points2, "../data/new_10.10.xlsx")
#     # get_two_line_chart(new_points1, new_points2)

if __name__ == '__main__':
    data1 = load_data("10.10.xlsx")
    points1 = get_points(data1)
    pd.DataFrame(points1).to_csv("../data/10.10.csv", index=False, header=0, sep=',')
