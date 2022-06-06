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
    start_time = parser.parse(data.iloc[0][0])
    trajectory = []
    for i in range(len(data)):
        trajectory.append(Point(x=data.iloc[i][1],
                                y=data.iloc[i][2],
                                t=(parser.parse(data.iloc[i][0]) - start_time).seconds))
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


if __name__ == '__main__':
    trajectories = get_berlin_mod_0_005_trajectories()
