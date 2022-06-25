import os.path
from dateutil import parser
import random

import numpy as np
import pandas as pd
from typing import List

from Experiment.common.Point import Point
from Experiment.common.Trajectory import Trajectory


def generate_trajectory(data: pd.DataFrame, is_random=False) -> List[Point]:
    """
    根据 data 生成一组带扰动的轨迹数据 或者 获得原始数据
    :param data: pd.DataFrame
    :param is_random: 是否随机扰动
    :return: Trajectory
    """
    trajectory = []
    for i in range(len(data)):
        if is_random:
            trajectory.append(Point(x=data.iloc[i][1] + random.uniform(-0.001, 0.001),
                                    y=data.iloc[i][2] + random.uniform(-0.001, 0.001),
                                    t=data.iloc[i][0] + random.randint(-5, 5)))
        else:
            trajectory.append(Point(x=data.iloc[i][1],
                                    y=data.iloc[i][2],
                                    t=data.iloc[i][0]))
    return trajectory


def load_trajectory(data: pd.DataFrame) -> List[Point]:
    accum_time = 0
    start_time = parser.parse(data.iloc[0][0])
    trajectory = []
    pre_point = Point(x=0, y=0, t=-1)
    for i in range(len(data)):
        cur_point = Point(x=data.iloc[i][1], y=data.iloc[i][2],
                          t=(parser.parse(data.iloc[i][0]) - start_time).seconds + accum_time)
        # 防止BerlinMOD_0_005Data数据集中 相同时间不同距离的点的问题
        if cur_point.t == pre_point.t:
            cur_point.t = pre_point.t + 1
        # 防止BerlinMOD_0_005Data数据集中 出现时间跳跃的问题
        elif abs(cur_point.x - pre_point.x) < 1e-6 and abs(cur_point.y - pre_point.y) < 1e-6:
            start_time = parser.parse(data.iloc[i][0])
            accum_time = pre_point.t
            cur_point.t = accum_time + 1

        trajectory.append(cur_point)
        pre_point = cur_point
    return trajectory


def get_trajectories(trajectory_type="trajectory"):
    """
    获得 生成的带扰动的7条相似 轨迹数据
    :param trajectory_type: trajectory 为 Trajectory 类型；point_list 为 list(points) 类型
    :return:
    """
    path = r'E:\Desktop\Programmer\PythonFile\PythonProject\Experiment\data\SyntheticData\output_origin_trajectory_'
    trajectories = []
    if trajectory_type == "trajectory":
        for i in range(7):
            trajectories.append(Trajectory(i, generate_trajectory(
                pd.read_csv(path + str(i) + '.csv', header=None))))
        return trajectories
    if trajectory_type == "point_list":
        for i in range(7):
            trajectories.append(generate_trajectory(
                pd.read_csv(path + str(i) + '.csv', header=None)))
    return trajectories


def get_berlin_mod_0_005_trajectories(trajectory_type="trajectory"):
    """
    获得 berlin_mod_0_005 筛选出的 10条相似轨迹
    :param trajectory_type: trajectory 为 Trajectory 类型；point_list 为 list(points) 类型
    :return:
    """
    path = r'E:\Desktop\Programmer\PythonFile\PythonProject\Experiment\data\BerlinMOD_0_005Data\select_trajectory_'
    trajectories = []
    if trajectory_type == "trajectory":
        for i in range(10):
            trajectories.append(Trajectory(i, load_trajectory(
                pd.read_csv(path + str(i + 1) + '.txt', header=None))))
        return trajectories
    if trajectory_type == "point_list":
        for i in range(10):
            trajectories.append(load_trajectory(
                pd.read_csv(path + str(i + 1) + '.txt', header=None)))
    return trajectories


def get_walk_data(trajectory_type="trajectory"):
    """
    获得 自己采集的行走 轨迹
    :param trajectory_type: trajectory 为 Trajectory 类型；point_list 为 list(points) 类型
    :return:
    """
    path = r'E:\Desktop\Programmer\PythonFile\PythonProject\Experiment\data\WalkingData'
    filenames = ["5-31-1", "5-31-2", "6-1-1", "6-1-2", "6-2-1", "6-2-2", "6-3-1", "6-3-2", "6-4-2"]
    trajectories = []
    if trajectory_type == "trajectory":
        for i in range(len(filenames)):
            trajectories.append(Trajectory(i, load_trajectory(
                pd.read_csv(path + "\\" + filenames[i] + '.txt', header=None, sep='\t'))))
        return trajectories
    if trajectory_type == "point_list":
        for i in range(len(filenames)):
            trajectories.append(load_trajectory(
                pd.read_csv(path + "\\" + filenames[i] + '.txt', header=None, sep='\t')))
    return trajectories


def get_airline_data(trajectory_type="trajectory"):
    """
    获得 飞机航线 轨迹
    :param trajectory_type: trajectory 为 Trajectory 类型；point_list 为 list(points) 类型
    :return:
    """
    path = r'E:\Desktop\Programmer\PythonFile\PythonProject\Experiment\data\AirlineData'
    trajectories = []
    trajectory_id = 0
    if trajectory_type == "trajectory":
        for i in range(9, 26):
            trajectories.append(Trajectory(trajectory_id, load_trajectory(
                pd.read_csv(path + "\\10." + str(i) + '.txt', header=None, sep='\t'))))
            trajectory_id += 1
        return trajectories
    if trajectory_type == "point_list":
        for i in range(9, 26):
            trajectories.append(load_trajectory(
                pd.read_csv(path + "\\10." + str(i) + '.txt', header=None, sep='\t')))
    return trajectories


if __name__ == '__main__':
    trajectories = get_airline_data()
