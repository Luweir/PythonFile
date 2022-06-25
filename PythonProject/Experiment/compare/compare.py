from typing import List

import pandas as pd
import math
import numpy as np
from fastdtw import fastdtw

import Experiment.FastDTW.fast_dtw_neigbors as dtw
import Experiment.Threshold.threshold as threshold
from Experiment.common.Point import Point


def get_PED_error(point: List[Point], sample: List[Point]) -> list:
    """
    返回压缩轨迹和原始轨迹的 PED 误差
    :param point: 原始轨迹
    :param sample: 简化轨迹
    :return: [平均误差，最大误差]
    """
    max_ped_error = 0
    left_point = sample[0]
    sample_index = 1
    right_point = sample[sample_index]
    point_index = 0
    sum_ped_error = 0
    while sample_index < len(sample):
        # 如果当前点在简化后的两点之间
        while left_point.t <= point[point_index].t < right_point.t:
            cur_point = point[point_index]
            # 计算PED误差 如果左右点位置相同 那么 就计算cur_point与他们的点的距离
            if left_point.x == right_point.x and left_point.y == right_point.y:
                ped_error = cur_point.distance(left_point)
            else:
                ped_error = cur_point.get_ped(left_point, right_point)
            sum_ped_error += ped_error
            max_ped_error = max(max_ped_error, ped_error)
            point_index += 1
            if point_index >= len(point):
                return [sum_ped_error / len(point), max_ped_error]
        sample_index += 1
        if sample_index >= len(sample):
            break
        left_point = right_point
        right_point = sample[sample_index]
    return [sum_ped_error / len(point), max_ped_error]


def get_SED_error(point: List[Point], sample: List[Point]) -> list:
    """
    返回压缩轨迹 和 原始轨迹的 SED 误差
    :param point: 原始轨迹
    :param sample: 简化轨迹
    :return: [平均SED误差，最大SED误差]
    """
    max_sed_error = 0
    left_point = sample[0]
    sample_index = 1
    right_point = sample[sample_index]
    point_index = 0
    sum_sed_error = 0
    while sample_index < len(sample):
        # 如果当前点在简化后的两点之间
        while left_point.t <= point[point_index].t < right_point.t:
            cur_point = point[point_index]
            # 计算SED误差 如果左右点位置相同 那么就计算cur_point与二者任一点的距离
            if left_point.x == right_point.x and left_point.y == right_point.y:
                sed_error = cur_point.distance(left_point)
            else:
                sed_error = cur_point.get_sed(left_point, right_point)
            sum_sed_error += sed_error
            max_sed_error = max(max_sed_error, sed_error)
            point_index += 1
            if point_index >= len(point):
                return [sum_sed_error / len(point), max_sed_error]
        sample_index += 1
        if sample_index >= len(sample):
            break
        left_point = right_point
        right_point = sample[sample_index]

    return [sum_sed_error / len(point), max_sed_error]


def get_speed_error(point: List[Point], sample: List[Point]) -> list:
    """
    获得原始轨迹与压缩轨迹的速度误差
    :param point:
    :param sample:
    :return:[平均速度误差，最大速度误差]
    """
    max_speed_error = 0
    left_point = sample[0]
    sample_index = 1
    right_point = sample[sample_index]
    point_index = 0
    sum_speed_error = 0
    while sample_index < len(sample):
        # 如果当前点在简化后的两点之间
        while point_index < len(point) - 1 and left_point.t <= point[point_index].t < right_point.t:
            # print("left_point", left_point.to_list())
            # print("right_point", right_point.to_list())
            # print("point[point_index]", point[point_index].to_list())
            # print("point[point_index+1]", point[point_index + 1].to_list())
            simulate_point1 = point[point_index].linear_prediction(left_point, right_point)
            simulate_point2 = point[point_index + 1].linear_prediction(left_point, right_point)
            # print("simulate_point1", simulate_point1.to_list())
            # print("simulate_point2", simulate_point2.to_list())
            if simulate_point1.t != simulate_point2.t:
                # 计算speed误差
                speed_error = abs(
                    point[point_index].get_speed(point[point_index + 1]) - simulate_point1.get_speed(simulate_point2))
                max_speed_error = max(max_speed_error, speed_error)
                sum_speed_error += speed_error
            point_index += 1
            if point_index >= len(point):
                return [sum_speed_error / len(point), max_speed_error]
        sample_index += 1
        if sample_index >= len(sample):
            break
        left_point = right_point
        right_point = sample[sample_index]
    return [sum_speed_error / len(point), max_speed_error]


def get_angle_error(point: List[Point], sample: List[Point]) -> list:
    """
    获得原始轨迹 与 压缩轨迹的角度误差
    :param point:
    :param sample:
    :return:[平均角度误差，最大角度误差]
    """
    max_angle_error = 0
    left_point = sample[0]
    sample_index = 1
    right_point = sample[sample_index]
    point_index = 0
    sum_angle_error = 0
    while sample_index < len(sample):
        sample_angle = math.atan2(right_point.y - left_point.y, right_point.x - left_point.x)
        while left_point.t <= point[point_index].t < right_point.t:
            cur_point = point[point_index]
            if cur_point.t != left_point.t:
                # 计算angle误差
                angle_error = abs(
                    abs(sample_angle) - abs(math.atan2(cur_point.y - left_point.y, cur_point.x - left_point.x)))
                max_angle_error = max(max_angle_error, angle_error)
                sum_angle_error += angle_error
            point_index += 1
            if point_index >= len(point):
                return [sum_angle_error / len(point), max_angle_error]
        sample_index += 1
        if sample_index >= len(sample):
            break
        left_point = right_point
        right_point = sample[sample_index]
    return [sum_angle_error / len(point), max_angle_error]


def get_haversine(point_a, point_b):
    EARTH_RADIUS = 6371229  # m 用于两点间距离计算
    lat1 = point_a[1] * math.pi / 180
    lat2 = point_b[1] * math.pi / 180
    lon1 = point_a[2] * math.pi / 180
    lon2 = point_b[2] * math.pi / 180
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a_a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a_a), math.sqrt(1 - a_a))
    return EARTH_RADIUS * c


def get_dtw(point: List[Point], sample: List[Point]):
    t1 = []
    t2 = []
    for ele in point:
        t1.append(ele.to_list())
    for ele in sample:
        t2.append(ele.to_list())
    distance, path = fastdtw(t1, t2, dist=get_haversine)
    return distance / 1000.0


if __name__ == '__main__':
    a = [Point(x=1.1, y=2.1, t=0), Point(x=1.5, y=2.1, t=2)]
    b = [Point(x=1.1, y=2.1, t=0), Point(x=1.3, y=2.1, t=1), Point(x=1.5, y=2.1, t=2)]
    print(get_dtw(a, b))
