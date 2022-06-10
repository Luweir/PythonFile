import time
import math

from typing import List

import Experiment.compare.compare as cr
import pandas as pd
from Experiment.common.Point import Point


# 加载数据集
def gps_reader(filename):
    points = []  # Track points set
    data = pd.read_csv("../data/" + filename, header=None, sep=',').values.tolist()
    for ele in data:
        points.append([ele[0], ele[1], ele[2]])
    return points


def get_ped(s, m, e):
    a = e[2] - s[2]
    b = s[1] - e[1]
    c = e[1] * s[2] - s[1] * e[2]
    if a == 0 and b == 0:
        return 0
    short_dist = abs((a * m[1] + b * m[2] + c) / math.sqrt(a * a + b * b))
    return short_dist


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


def douglas_peucker(points: List[Point], start: int, last: int, epsilon: float):
    """
        DP 算法
    :param points:
    :param start:
    :param last:
    :param epsilon: 误差阈值  epsilon为欧氏距离   epsilon*100000≈实际距离
    :return: 简化后的索引序列
    """
    d_max = 0
    index = start
    rec_result = []
    for i in range(start + 1, last):
        d = points[i].get_ped(points[start], points[last])
        if d > d_max:
            index = i
            d_max = d
    if d_max > epsilon:
        rec_result1 = douglas_peucker(points, start, index, epsilon)
        rec_result2 = douglas_peucker(points, index, last, epsilon)
        rec_result.extend(rec_result1)
        rec_result.extend(rec_result2[1:])
    else:
        rec_result.append(start)
        rec_result.append(last)
    return rec_result


def new_dp(points: List[Point], start: int, last: int, epsilon: float = 0.2):
    """
        改进的 DP 算法  综合考虑速度、方向和距离误差   每个段的 sed 误差阈值都不一样
    :param points:
    :param start:
    :param last:
    :param epsilon: 误差容忍度  浮点数 0-1范围内
    :return: 简化后的索引序列
    """
    seg_speed = points[start].get_speed(points[last])
    # 该段的误差距离阈值
    d_threshold = seg_speed * (1 + epsilon) * math.sin(epsilon * 2)
    # 最大sed误差阈值 和 索引
    d_max = 0
    index = start
    # 满足速度和方向容忍度表示
    tol_d_max = 0
    tol_index = start
    rec_result = []
    for i in range(start + 1, last):
        d = points[i].get_ped(points[start], points[last])
        if d > d_max:
            index = i
            d_max = d
        if points[i - 1].t == points[i].t:
            print(points[i].t, points[i].x, points[i].y)
            print(points[i - 1].t, points[i - 1].x, points[i - 1].y)
        v = points[i].get_speed(points[i - 1])
        a = points[i].get_angle(points[start], points[last])
        if v < seg_speed * (1 - epsilon) or v > seg_speed * (1 + epsilon) or a > 2 * epsilon:
            if d > tol_d_max:
                tol_d_max = d
                tol_index = i
    if d_max > d_threshold:
        rec_result1 = douglas_peucker(points, start, index, epsilon)
        rec_result2 = douglas_peucker(points, index, last, epsilon)
        rec_result.extend(rec_result1)
        rec_result.extend(rec_result2[1:])
    elif tol_d_max != 0:
        rec_result1 = douglas_peucker(points, start, tol_index, epsilon)
        rec_result2 = douglas_peucker(points, tol_index, last, epsilon)
        rec_result.extend(rec_result1)
        rec_result.extend(rec_result2[1:])
    else:
        rec_result.append(start)
        rec_result.append(last)
    return rec_result


# TD-TR
def td_tr(points: List[Point], start: int, last: int, epsilon: float) -> list:
    """
    td-tr 算法
    :param points:
    :param start:
    :param last:
    :param epsilon:
    :return:
    """
    d_max = 0
    index = start
    rec_result = []
    for i in range(start + 1, last):
        d = points[i].get_sed(points[start], points[last])
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


if __name__ == '__main__':
    print(math.sin(30 / 180 * math.pi))
    print(math.asin(1 / 2), "", 30 / 180 * math.pi)
