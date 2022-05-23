import copy

import numpy as np
import pandas as pd
import math

from typing import List

from Experiment.TrajStore.src.cluster import traj_store_cluster
import random

from Experiment.common.Point import Point
from Experiment.common.Trajectory import Trajectory
from Experiment.compare.compare import get_PED_error


def generate_trajectory(data):
    trajectory = []
    for i in range(len(data)):
        # trajectory.append(Point(data.iloc[i][1] + random.uniform(-0.001, 0.001),
        #                         data.iloc[i][2] + random.uniform(-0.001, 0.001),
        #                         data.iloc[i][0] + random.randint(-5, 5)))
        trajectory.append(Point(x=data.iloc[i][1],
                                y=data.iloc[i][2],
                                t=data.iloc[i][0]))
    return trajectory


def output_origin_trajectory(trajectories):
    for trajectory in trajectories:
        data = []
        start_time = trajectory.points[0].t
        for point in trajectory.points:
            data.append([point.t - start_time, round(point.x, 4), round(point.y, 4)])
        pd.DataFrame(data).to_csv("output_origin_trajectory_" + str(trajectory.trajectory_id) + ".csv", header=False,
                                  index=False)


def output_compressed_trajectory(trajectories):
    data = []
    for trajectory in trajectories:
        # 自身就是本聚类里面的参考轨迹
        if trajectory.reference_trajectory_id == -1:
            for point in trajectory.points:
                data.append([int(point.t), round(point.x, 4), round(point.y, 4)])
        # 有参考轨迹 就只用存时间对
        else:
            for i in range(len(trajectory.points)):
                point = trajectory.points[i]
                data.append([int(point.t), int(trajectory.reference_time[i])])
                i += 1
    pd.DataFrame(data).to_csv("output_compressed_trajectory.txt", header=False,
                              index=False)


def get_traj_store_ped_error(origin_trajectory: Trajectory, compressed_trajectory: Trajectory, hash_map):
    """
    计算压缩后轨迹的ped误差
    :param origin_trajectory: 原始轨迹
    :param compressed_trajectory: 压缩后的轨迹
    :param hash_map: 参考轨迹映射 <trajectory_id,trajectory>
    :return: [average_ped_error,max_ped_error]
    """
    # 无参考轨迹
    if compressed_trajectory.reference_trajectory_id == -1:
        return get_PED_error(origin_trajectory.to_list(), compressed_trajectory.to_list())
    # 有参考轨迹 则进行轨迹恢复   先把参考的点给恢复，然后再进行ped误差计算
    reference_trajectory = hash_map[compressed_trajectory.reference_trajectory_id]
    for i in range(len(compressed_trajectory.reference_time)):
        match_time = compressed_trajectory.reference_time[i]
        # 匹配时间超过参考轨迹最大时间 说明是 最后一个点到第一个点之间
        if match_time > reference_trajectory.points[-1].t:
            estimation_point = \
                reference_trajectory.points[-1].get_point_by_time_and_line(match_time, reference_trajectory.points[0])
            compressed_trajectory.points[i].x = estimation_point.x
            compressed_trajectory.points[i].y = estimation_point.y
            continue
        for j in range(len(reference_trajectory.points) - 1):
            if reference_trajectory.points[j].t <= match_time <= reference_trajectory.points[j + 1].t:
                pre_point = reference_trajectory.points[j]
                next_point = reference_trajectory.points[j + 1]
                estimation_point = pre_point.get_point_by_time_and_line(match_time, next_point)
                # 根据映射时间恢复位置
                compressed_trajectory.points[i].x = estimation_point.x
                compressed_trajectory.points[i].y = estimation_point.y
                break
    return get_PED_error(origin_trajectory.to_list(), compressed_trajectory.to_list())

# 这里有问题
def linear_eliminate(trajectories: List[Trajectory], epsilon: float):
    """
    对 trajectories 进行线性消除无关点
    :param trajectories: 轨迹集
    :param epsilon: 欧氏距离误差 约等于 实际距离/100000
    :return: 返回进行线性消除后的轨迹集
    """
    linear_eliminate_trajectories = []
    for trajectory in trajectories:
        new_trajectory_points = []
        first_index = 0
        last_index = first_index + 1
        while last_index < len(trajectory.points):
            first_point = trajectory.points[first_index]
            last_point = trajectory.points[last_index]
            flag = True
            for mid_index in range(first_index + 1, last_index):
                mid_point = trajectory.points[mid_index]
                temp_point = mid_point.linear_prediction(first_point, last_point)
                if mid_point.distance(temp_point) > epsilon:
                    flag = False
                    break
            if not flag or last_index == len(trajectory.points) - 1:
                new_trajectory_points.append(first_point)
                first_index = last_index
            last_index += 1
        # 加入最后一个点
        new_trajectory_points.append(trajectory.points[-1])
        linear_eliminate_trajectories.append(Trajectory(trajectory.trajectory_id, new_trajectory_points))
    return linear_eliminate_trajectories


if __name__ == '__main__':
    # data1 = pd.read_csv("../../data/10.9.csv", header=None)
    origin_trajectories = [
        Trajectory(0, generate_trajectory(pd.read_csv("output_origin_trajectory_0.csv", header=None))),
        Trajectory(1, generate_trajectory(pd.read_csv("output_origin_trajectory_1.csv", header=None))),
        # Trajectory(2, generate_trajectory(pd.read_csv("output_origin_trajectory_2.csv", header=None))),
        # Trajectory(3, generate_trajectory(pd.read_csv("output_origin_trajectory_3.csv", header=None))),
        # Trajectory(4, generate_trajectory(pd.read_csv("output_origin_trajectory_4.csv", header=None))),
        # Trajectory(5, generate_trajectory(pd.read_csv("output_origin_trajectory_5.csv", header=None))),
        # Trajectory(6, generate_trajectory(pd.read_csv("output_origin_trajectory_6.csv", header=None)))
    ]
    # trajectories = [Trajectory(0, generate_trajectory(data1)),
    #                 Trajectory(1, generate_trajectory(data1)),
    #                 Trajectory(2, generate_trajectory(data1)),
    #                 Trajectory(3, generate_trajectory(data1)),
    #                 Trajectory(4, generate_trajectory(data1)),
    #                 Trajectory(5, generate_trajectory(data1)),
    #                 Trajectory(6, generate_trajectory(data1))]
    # print(trajectories)
    epsilon = 200
    # 第一部分 线性法消除无关点
    linear_eliminate_trajectories = linear_eliminate(origin_trajectories, epsilon / 100000.0)
    # linear_eliminate_trajectories = copy.deepcopy(origin_trajectories)
    # 第二部分 聚类压缩
    group = traj_store_cluster(linear_eliminate_trajectories, epsilon)

    # output_origin_trajectory(trajectories)
    output_compressed_trajectory(linear_eliminate_trajectories)

    hash_map = {}
    for trajectory in linear_eliminate_trajectories:
        if trajectory.reference_trajectory_id == -1:
            hash_map[trajectory.trajectory_id] = trajectory
    total_ped = 0
    max_ped_error = 0
    for i in range(len(linear_eliminate_trajectories)):
        [a, b] = get_traj_store_ped_error(origin_trajectories[i], linear_eliminate_trajectories[i], hash_map)
        total_ped += a
        max_ped_error = max(max_ped_error, b)
    print("average ped error:", total_ped / len(origin_trajectories))
    print("max ped error:", max_ped_error)
