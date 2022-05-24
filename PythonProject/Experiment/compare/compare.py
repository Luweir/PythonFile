from typing import List

import pandas as pd
import math
import numpy as np
import Experiment.FastDTW.fast_dtw_neigbors as dtw
import Experiment.Threshold.threshold as threshold
from Experiment.common.Point import Point


def get_CR_and_time(save_filename, start_time, end_time, points, sample):
    # first_point_time = sample[0][0]
    # for e in sample:
    #     e[0] = e[0] - first_point_time
    # pd.DataFrame(sample).to_csv(save_filename, index=False, header=0, sep=',')
    print("run time: " + str(end_time - start_time))
    print("before: " + str(len(points)) + " after: " + str(len(sample)))
    print("compress ratio: " + str(len(points) / len(sample)))


def get_PED_error(point: List[Point], sample: List[Point]):
    max_ped_error = 0
    left_point = sample[0]
    sample_index = 1
    right_point = sample[sample_index]
    point_index = 0
    sum_ped_error = 0
    while sample_index < len(sample):
        # 如果当前点在简化后的两点之间
        while left_point.t <= point[point_index].t <= right_point.t:
            if point_index < len(point) and point[point_index].t == right_point.t:
                break
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


def get_SED_error(point, sample):
    left_point = sample[0]
    sample_index = 1
    right_point = sample[sample_index]
    point_index = 0
    sed_error = 0
    while sample_index < len(sample):
        if right_point[1] != left_point[1] and right_point[2] != left_point[2]:
            # 如果当前点在简化后的两点之间
            while left_point[0] <= point[point_index][0] <= right_point[0]:
                cur_point = point[point_index]
                # 计算SED误差
                new_x = left_point[1] + (right_point[1] - left_point[1]) * (cur_point[0] - left_point[0]) / (
                        right_point[0] - left_point[0])
                new_y = left_point[2] + (right_point[2] - left_point[2]) * (cur_point[0] - left_point[0]) / (
                        right_point[0] - left_point[0])
                sed_error += math.sqrt((cur_point[1] - new_x) ** 2 + (cur_point[2] - new_y) ** 2)
                point_index += 1
                if point_index >= len(point):
                    return sed_error / len(point)
        sample_index += 1
        if sample_index >= len(sample):
            return sed_error / len(point)
        left_point = right_point
        right_point = sample[sample_index]
    return sed_error / len(point)


# p2的速度误差 = p2的平均速度 - p1p3的平均速度
def get_speed_error(point, sample):
    left_point = sample[0]
    sample_index = 1
    right_point = sample[sample_index]
    point_index = 0
    speed_error = 0
    while sample_index < len(sample):
        if right_point[1] != left_point[1] and right_point[2] != left_point[2]:
            # 如果当前点在简化后的两点之间
            sample_speed = threshold.get_speed(left_point, right_point)
            while left_point[0] <= point[point_index][0] <= right_point[0]:
                # 计算speed误差
                point_speed = sample_speed
                if 0 < point_index < len(point) - 1:
                    point_speed1 = threshold.get_speed(point[point_index - 1], point[point_index])
                    point_speed2 = threshold.get_speed(point[point_index], point[point_index + 1])
                    point_speed = (point_speed1 + point_speed2) / 2
                speed_error += abs(sample_speed - point_speed)
                point_index += 1
                if point_index >= len(point):
                    return speed_error / len(point)
        sample_index += 1
        if sample_index >= len(sample):
            return speed_error / len(point)
        left_point = right_point
        right_point = sample[sample_index]
    return speed_error / len(point)


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


if __name__ == '__main__':
    # print(get_SED_error([[2, 3.5, 6.1]], [[0, 3.8, 4.2], [3, 3.9, 5.1]]))
    a = [[1, 2, 3], [2, 3, 4]]
    print(np.array(a)[:, 1:3].tolist())
    ...
