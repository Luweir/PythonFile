import numpy as np
import pandas as pd
import time
import math
import Experiment.compare.compare as cr


# points=[点1(时间time,纬度lat,经度lon),点2(时间time,纬度lat,经度lon),...]
# 加载数据集
def gps_reader(filename):
    points = []  # Track points set
    data = pd.read_csv("../data/" + filename, header=None, sep=',').values.tolist()
    for ele in data:
        points.append([ele[0], ele[1], ele[2]])
    return points


def get_haversine(point_a, point_b):
    lat1 = point_a[1] * math.pi / 180
    lat2 = point_b[1] * math.pi / 180
    lon1 = point_a[2] * math.pi / 180
    lon2 = point_b[2] * math.pi / 180
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a_a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a_a), math.sqrt(1 - a_a))
    return EARTH_RADIUS * c


# 计算两点之间的距离
def get_distance(point_a, point_b):
    return get_haversine(point_a, point_b)  # 距离单位：米


# 获得两点之间的速度
def get_speed(point_a, point_b):
    return get_distance(point_a, point_b) / (point_b[0] - point_a[0])


# 获得两点之间的角度
def get_angel(point_a, point_b):
    lat_diff = point_b[1] - point_a[1]
    lon_diff = point_b[2] - point_a[2]
    return math.atan2(lon_diff, lat_diff)  # 得到的是弧度制， x*Math.PI/180 => 角度


# To determine whether a point e at a safe angle area
def safe_angle(sample_b, sample_c, point_c, point_d, point_e):
    angel_sample_bc = get_angel(sample_b, sample_c)
    angle_de = get_angel(point_d, point_e)
    angle_trajectory_cd = get_angel(point_c, point_d)
    return abs(angle_de - angle_trajectory_cd) <= angle_threshold and abs(angle_de - angel_sample_bc) <= angle_threshold


# 检查速度是否在范围内
def safe_speed(sample_b, sample_c, point_c, point_d, point_e):
    sample_speed = get_speed(sample_b, sample_c)
    trajectory_speed = get_speed(point_c, point_d)
    de_speed = get_speed(point_d, point_e)
    # Sample speed and actual trajectory is within the scope of the threshold value
    return abs(trajectory_speed - de_speed) <= speed_threshold and abs(sample_speed - de_speed) <= speed_threshold


# 核心函数
def threshold(points):
    sample = [points[0], points[1]]
    for i in range(2, len(points) - 1):
        if safe_speed(sample[len(sample) - 2], sample[len(sample) - 1], points[i - 2], points[i - 1],
                      points[i]) and safe_angle(sample[len(sample) - 2], sample[len(sample) - 1], points[i - 2],
                                                points[i - 1], points[i]):
            continue
        else:
            sample.append(points[i])
    # 最后一个元素放入
    sample.append(points[len(points) - 1])
    return sample


EARTH_RADIUS = 6371229  # m 用于两点间距离计算

# 阈值设定
speed_threshold = 20  # 23m/s => 80km/h CR=3.2；2.3m/s => 8km/h CR=2.96   origin value=1.5
angle_threshold = 0.2  # 0.8 => 45° CR=3.2 ；0.6 => 34°s CR=2.72

if __name__ == '__main__':
    filename = "10.9.csv"
    save_filename = "result.csv"
    points = gps_reader(filename)
    start_time = time.perf_counter()  # Note: python3.8 does not support time.clock()
    sample = threshold(points)
    end_time = time.perf_counter()
    print("Threshold")
    print("dataset:" + str(filename))
    cr.get_CR_and_time(save_filename, start_time, end_time, points, sample)
    print("PED距离误差：" + str(cr.get_PED_error(points, sample)))
    print("SED距离误差：" + str(cr.get_SED_error(points, sample)))
    print("Angle角度误差：" + str(cr.get_angle_error2(points, sample)))
    print("Speed速度误差：" + str(cr.get_speed_error(points, sample)))
    cr.get_dtw(points, sample)
