import math
import time
import pandas as pd
import numpy as np
import Experiment.compare.compare as cr


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


# __base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
# __decodemap = {}
# for i in range(len(__base32)):
#     __decodemap[__base32[i]] = i


# 用粗粒度位图得到的能够粗略表示该轨迹的整数
# （即使是超出 经纬度范围 也可以用这个  因为这个是存储的大多周期性轨迹常经过的位置）
def grid_bitmap(trajectory, lat_min, lat_max, lon_min, lon_max, row=10, col=10):
    diff_lat = (lat_max - lat_min) / row
    diff_lon = (lon_max - lon_min) / col
    # 初始化位图   横8纵8
    bitmap = np.zeros((row, col))
    # 是均等分还是 根据 GeoHash 交叉分 ? 均等分能够尽可能地保留原始轨迹曲线
    # 位图中进行位置标记

    # 先直接用整数存  因为python的整数无限容量
    geohash = 0
    for point in trajectory:
        # floor((lat-lat_min)/diff_lat)  and  floor((lon-lon_min)/diff_lon)
        i = math.floor((point[1] - lat_min) / diff_lat)
        j = math.floor((point[2] - lon_min) / diff_lon)
        bitmap[i, j] = 1

        # 0000 0000
        # 1000 0000
        # => 1000 0000 0000 0000
        geohash += (1 << (i * row + col - 1 - j))
    return geohash


latitude_min = 90
latitude_max = -90
longitude_min = 180
longitude_max = -180


# 获取轨迹经纬度的范围
def set_range(points):
    points = np.array(points)
    # 找到轨迹经纬度上下限   并对上限取上界  对下限取下界
    global latitude_min, latitude_max, longitude_min, longitude_max
    latitude_min = math.floor(np.array(points[:, 1]).min())
    latitude_max = math.ceil(np.array(points[:, 1]).max())
    longitude_min = math.floor(np.array(points[:, 2]).min())
    longitude_max = math.ceil(np.array(points[:, 2]).max())


# 统计geoHash 匹配1的个数
def match(cur_trajectoryHash, conf_trajectoryHash):
    return 1


def check_in(cur_point, left_point, right_point):
    flag1 = False
    flag2 = False
    if left_point[1] >= cur_point[1] >= right_point[1] or left_point[1] <= cur_point[1] <= right_point[1]:
        flag1 = True
    if left_point[2] >= cur_point[2] >= right_point[2] or left_point[2] <= cur_point[2] <= right_point[2]:
        flag1 = True
    return flag1 and flag2


# 多轨迹压缩 运行
def run_mtc(trajectory, refe_trajectory):
    # 搜索找到最相似的轨迹
    # similar_id = -1
    # most_match = 0
    # for trajectory_id in trajectory_hash.keys():
    #     cur_match = match(cur_trajectoryHash, trajectory_hash[trajectory_id])
    #     if cur_match > most_match:
    #         similar_id = trajectory_id
    #         most_match = cur_match

    # 多项式拟合  得到对应 (t,t') (p,p')
    for i in range(len(trajectory)):
        cur_point = trajectory[i]
        ref_point = []
        for j in range(len(refe_trajectory) - 1):
            # 找到经过 cur_point 的位置
            if check_in(cur_point, refe_trajectory[j], refe_trajectory[j + 1]):
                # 进行多项式拟合
                trajectory_seg = refe_trajectory[j - 2 if j >= 0 else 0:j + 3 if j + 3 < len(refe_trajectory) else len(
                    refe_trajectory) - 1]
                trajectory_seg = np.array(trajectory_seg)
                z1 = np.polyfit(trajectory_seg[:, 0], trajectory_seg[:, 1], 2)  # 用n次多项式拟合，可改变多项式阶数；
                p1 = np.poly1d(z1)  # 得到多项式系数，按照阶数从高到低排列
                
    ...


# ID trajectoryHash
trajectory_hash = {}
# ID trajectory
trajectories = {}

if __name__ == '__main__':
    filename = "10.9.csv"
    epsilon = 0.001
    save_filename = "result.csv"
    points = gps_reader(filename)
    # 一、设置经纬度上下限
    # set_range(points)

    # 二、STC -  td-tr 算法压缩
    start_time = time.perf_counter()
    # sample_index = td_tr(points, 0, len(points) - 1, epsilon)
    # sample = []
    # for e in sample_index:
    #     sample.append(points[e])
    end_time = time.perf_counter()
    # sample = np.array(sample)

    # # 三、对当前轨迹进行相似轨迹粗粒度搜寻 然后MTC压缩
    # print("DP-TR")
    # print("dataset:" + str(filename))
    # cr.get_CR_and_time(save_filename, start_time, end_time, points, sample)
    # print("PED距离误差：" + str(cr.get_PED_error(points, sample)))
    # print("SED距离误差：" + str(cr.get_SED_error(points, sample)))
    # print("Angle角度误差：" + str(cr.get_angle_error2(points, sample)))
    # # cr.get_dtw(points, sample)
    gb1 = grid_bitmap(points, latitude_min, latitude_max, longitude_min, longitude_max)
    print(gb1)
    # # 简化后的与简化前的有一些不同  可能是中间有些点被忽略   可以通过差值把忽略的格子给设置为1
    # gb1 = grid_bitmap(sample, latitude_min, latitude_max, longitude_min, longitude_max)
    # print(gb1)
    # points = gps_reader("10.10.csv")
    # gb2 = grid_bitmap(points, latitude_min, latitude_max, longitude_min, longitude_max)
    # points = gps_reader("10.11.csv")
    # gb3 = grid_bitmap(points, latitude_min, latitude_max, longitude_min, longitude_max)
    # 1908598440434674518018754863244
    # 5076797241849641349311480676534
    # 6026915919326921468891933950092
