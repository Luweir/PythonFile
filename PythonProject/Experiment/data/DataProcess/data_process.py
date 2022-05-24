import os.path
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
            trajectory.append(Point(data.iloc[i][1] + random.uniform(-0.001, 0.001),
                                    data.iloc[i][2] + random.uniform(-0.001, 0.001),
                                    data.iloc[i][0] + random.randint(-5, 5)))
        else:
            trajectory.append(Point(x=data.iloc[i][1],
                                    y=data.iloc[i][2],
                                    t=data.iloc[i][0]))
    return trajectory


def get_trajectories(trajectory_type="trajectory"):
    """
    获得原始轨迹数据
    :param trajectory_type: trajectory 为 Trajectory 类型；point_list 为 list(points) 类型
    :return:
    """
    path = r'E:\Desktop\Programmer\PythonFile\PythonProject\Experiment\data\SyntheticData'
    trajectories = []
    if trajectory_type == "trajectory":
        for i in range(7):
            trajectories.append(Trajectory(i, generate_trajectory(
                pd.read_csv(path + r'\output_origin_trajectory_' + str(i) + '.csv', header=None))))
        return trajectories
    if trajectory_type == "point_list":
        for i in range(7):
            trajectories.append(generate_trajectory(
                pd.read_csv(path + r'\output_origin_trajectory_' + str(i) + '.csv', header=None)))
    return trajectories
