from typing import List

import pandas as pd
import math
import numpy as np
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
            # if point_index < len(point) and point[point_index].t == right_point.t:
            #     break
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


# p2的速度误差 = p2的平均速度 - p1p3的平均速度
def get_speed_error(point: List[Point], sample: List[Point]) -> list:
    """
    获得原始轨迹与压缩轨迹的速度误差
    :param point:
    :param sample:
    :return:
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
            simulate_point1 = point[point_index].linear_prediction(left_point, right_point)
            simulate_point2 = point[point_index + 1].linear_prediction(left_point, right_point)
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


# p2的角度误差 = p1p2的角度 与 p1p2'的角度差
def get_angle_error(point, sample):
    left_point = sample[0]
    sample_index = 1
    right_point = sample[sample_index]
    point_index = 0
    angle_error = 0
    while sample_index < len(sample):
        if right_point[1] != left_point[1] and right_point[2] != left_point[2]:
            lat_diff = right_point[1] - left_point[1]
            lon_diff = right_point[2] - left_point[2]
            sample_angle = math.atan2(lon_diff, lat_diff)
            # 如果当前点在简化后的两点之间
            while left_point[0] <= point[point_index][0] <= right_point[0]:
                cur_point = point[point_index]
                # 计算angle误差
                lat_diff = cur_point[1] - left_point[1]
                lon_diff = cur_point[2] - left_point[2]
                angle_error += abs(sample_angle - math.atan2(lon_diff, lat_diff))
                point_index += 1
                if point_index >= len(point):
                    return angle_error
        sample_index += 1
        if sample_index >= len(sample):
            return angle_error
        left_point = right_point
        right_point = sample[sample_index]
    return angle_error


def get_angle_error2(point, sample):
    left_point = sample[0]
    sample_index = 1
    right_point = sample[sample_index]
    point_index = 0
    angle_error = 0
    while sample_index < len(sample):
        if right_point[1] != left_point[1] and right_point[2] != left_point[2]:
            lat_diff = right_point[1] - left_point[1]
            lon_diff = right_point[2] - left_point[2]
            sample_angle = math.atan2(lon_diff, lat_diff)
            # 如果当前点在简化后的两点之间
            pre_point = left_point
            while left_point[0] <= point[point_index][0] <= right_point[0]:
                cur_point = point[point_index]
                # 计算angle误差
                lat_diff = cur_point[1] - pre_point[1]
                lon_diff = cur_point[2] - pre_point[2]
                angle_error += abs(sample_angle - math.atan2(lon_diff, lat_diff))
                point_index += 1
                pre_point = cur_point
                if point_index >= len(point):
                    return angle_error / len(point)
        sample_index += 1
        if sample_index >= len(sample):
            return angle_error / len(point)
        left_point = right_point
        right_point = sample[sample_index]
    return angle_error / len(point)


def get_dtw(point, sample):
    traj1 = np.array(point)
    traj2 = np.array(sample)
    return dtw.get_fastdtw(traj1, traj2)
