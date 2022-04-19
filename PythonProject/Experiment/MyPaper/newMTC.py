import math
import time
import pandas as pd
import numpy as np
import Experiment.compare_result.compare as cr


# 加载数据集
def gps_reader(filename):
    points = []  # Track points set
    data = pd.read_csv("../data/" + filename, header=None, sep=',').values.tolist()
    for ele in data:
        points.append([ele[0], ele[1], ele[2]])
    return points


# 获取时间同步距离
def get_sed(s, m, e):
    numerator = m[0] - s[0]
    denominator = e[0] - s[0]
    time_ratio = 1
    if denominator != 0:
        time_ratio = numerator / denominator
    lat = s[1] + (e[1] - s[1]) * time_ratio
    lon = s[2] + (e[2] - s[2]) * time_ratio
    lat_diff = lat - m[1]
    lon_diff = lon - m[2]
    return math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)


# TD-TR算法
def td_tr(points, start, last, epsilon):
    d_max = 0
    index = start
    rec_result = []
    for i in range(start + 1, last):
        d = get_sed(points[start], points[i], points[last])
        if d > d_max:
            index = i
            d_max = d
    if d_max > epsilon:
        rec_result1 = td_tr(points, start, index, epsilon)
        rec_result2 = td_tr(points, index, last, epsilon)
        rec_result.extend(rec_result1)
        # the first point must can't join because it equals with the last point of rec_result1
        rec_result.extend(rec_result2[1:])
    else:
        rec_result.append(start)
        rec_result.append(last)
    return rec_result


# 用粗粒度位图得到的（8*8）64位表示该轨迹
# （即使是超出 经纬度范围 也可以用这个  因为这个是存储的大多周期性轨迹常经过的位置）
def grid_bitmap(trajectory, lat_min, lat_max, lon_min, lon_max):
    lat_range = np.linspace(lat_min, lat_max, 8)
    lon_range = np.linspace(lon_min, lon_max, 8)
    diff_lat = (lat_max - lat_min) / 8
    diff_lon = (lon_max - lon_min) / 8
    # 初始化位图
    bitmap = np.zeros((8, 8))
    for point in trajectory:
        # floor((lat-lat_min)/diff_lat)  and  floor((lon-lon_min)/diff_lon)
        i = math.floor((point[1] - lat_min) / diff_lat)
        j = math.floor((point[2] - lon_min) / diff_lon)
        bitmap[i, j] = 1


if __name__ == '__main__':
    filename = "10.9.csv"
    epsilon = 0.01  # 0.0001
    save_filename = "result.csv"
    points = gps_reader(filename)

    # STC -  td-tr 算法压缩
    start_time = time.perf_counter()
    sample_index = td_tr(points, 0, len(points) - 1, epsilon)
    sample = []
    for e in sample_index:
        sample.append(points[e])
    end_time = time.perf_counter()

    # 找到轨迹经纬度上下限   并取上界和下界
    sample = np.array(sample)
    latitude_min = math.floor(np.array(sample[:, 1]).min())
    latitude_max = math.ceil(np.array(sample[:, 1]).max())
    longitude_min = math.floor(np.array(sample[:, 2]).min())
    longitude_max = math.ceil(np.array(sample[:, 2]).max())

    print("DP-TR")
    print("dataset:" + str(filename))
    cr.get_CR_and_time(save_filename, start_time, end_time, points, sample)
    print("PED距离误差：" + str(cr.get_PED_error(points, sample)))
    print("SED距离误差：" + str(cr.get_SED_error(points, sample)))
    print("Angle角度误差：" + str(cr.get_angle_error2(points, sample)))
    cr.get_dtw(points, sample)
