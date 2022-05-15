import numpy as np
import pandas as pd
import math
from Experiment.TrajStore.src.cluster import trajStore_cluster
from Experiment.TrajStore.src.Point import Point
from Experiment.TrajStore.src.Trajectory import Trajectory
import random


def generate_trajectory(data):
    trajectory = []
    for i in range(len(data)):
        # trajectory.append(Point(data.iloc[i][1] + random.uniform(-0.001, 0.001),
        #                         data.iloc[i][2] + random.uniform(-0.001, 0.001),
        #                         data.iloc[i][0] + random.randint(-5, 5)))
        trajectory.append(Point(data.iloc[i][1],
                                data.iloc[i][2],
                                data.iloc[i][0]))
    return trajectory


def output_origin_trajectory(trajectories):
    for trajectory in trajectories:
        data = []
        start_time = trajectory.points[0].t
        for point in trajectory.points:
            data.append([point.t - start_time, round(point.x, 4), round(point.y, 4)])
        pd.DataFrame(data).to_csv("output_origin_trajectory_" + str(trajectory.traj_id) + ".csv", header=False,
                                  index=False)


def output_compressed_trajectory(trajectories):
    data = []
    for trajectory in trajectories:
        start_time = trajectory.points[0].t
        # 自身就是本聚类里面的参考轨迹
        if trajectory.refe_traj_id == -1:
            for point in trajectory.points:
                data.append([point.t - start_time, round(point.x, 4), round(point.y, 4)])
        # 有参考轨迹 就只用存时间对
        else:
            refe_start_time = 0
            for traj in trajectories:
                if traj.traj_id == trajectory.refe_traj_id:
                    refe_start_time = traj.points[0].t
                    break
            i = 0
            while i < trajectory.traj_size:
                data.append([trajectory.points[i].t - start_time,
                             (int(trajectory.refe_time[i]) - refe_start_time) if i < len(trajectory.refe_time) else 0])
                i += 1
    pd.DataFrame(data).to_csv("output_compressed_trajectory.csv", header=False,
                              index=False)


# 两点之间的距离  欧氏距离
def euc_dist(p1, p2):
    return round(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 5)


max_ped_error = 0


def restore_trajectory_ped_error(trajectory, hash_map):
    global max_ped_error
    ped_error = 0
    if trajectory.refe_traj_id == -1:
        return 0
    refe_trajectory = hash_map[trajectory.refe_traj_id]
    restore_trajectory = []
    for i in range(len(trajectory.refe_time)):
        cur_time = trajectory.points[i].t
        match_time = trajectory.refe_time[i]
        for j in range(len(refe_trajectory.points) - 1):
            if refe_trajectory.points[j].t <= match_time <= refe_trajectory.points[j + 1].t:
                pre_point = refe_trajectory.points[j]
                next_point = refe_trajectory.points[j + 1]
                estimation_x = pre_point.x + (next_point.x - pre_point.x) * (match_time - pre_point.t) / (
                        next_point.t - pre_point.t)
                estimation_y = pre_point.y + (next_point.y - pre_point.y) * (match_time - pre_point.t) / (
                        next_point.t - pre_point.t)
                max_ped_error = max(max_ped_error,
                                    euc_dist(trajectory.points[i].to_list(), [estimation_x, estimation_y]))
                ped_error += euc_dist(trajectory.points[i].to_list(), [estimation_x, estimation_y])
                restore_trajectory.append([cur_time, estimation_x, estimation_y])
                break
    return ped_error / len(trajectory.points)


if __name__ == '__main__':
    data1 = pd.read_csv("../../data/10.9.csv", header=None)[:-10]
    # data2 = pd.read_csv("../../data/10.9.csv", header=None)
    # data3 = pd.read_csv("../../data/10.11.csv", header=None)
    trajectories = [Trajectory(0, generate_trajectory(pd.read_csv("output_origin_trajectory_0.csv", header=None))),
                    Trajectory(1, generate_trajectory(pd.read_csv("output_origin_trajectory_1.csv", header=None))),
                    Trajectory(2, generate_trajectory(pd.read_csv("output_origin_trajectory_2.csv", header=None))),
                    Trajectory(3, generate_trajectory(pd.read_csv("output_origin_trajectory_3.csv", header=None))),
                    Trajectory(4, generate_trajectory(pd.read_csv("output_origin_trajectory_4.csv", header=None))),
                    Trajectory(5, generate_trajectory(pd.read_csv("output_origin_trajectory_5.csv", header=None))),
                    Trajectory(6, generate_trajectory(pd.read_csv("output_origin_trajectory_6.csv", header=None)))]
    # trajectories = [Trajectory(0, generate_trajectory(data1)),
    #                 Trajectory(1, generate_trajectory(data1)),
    #                 Trajectory(2, generate_trajectory(data1)),
    #                 Trajectory(3, generate_trajectory(data1)),
    #                 Trajectory(4, generate_trajectory(data1)),
    #                 Trajectory(5, generate_trajectory(data1)),
    #                 Trajectory(6, generate_trajectory(data1))]
    print(trajectories)
    group = trajStore_cluster(trajectories, 500)
    # print(group)
    output_origin_trajectory(trajectories)
    output_compressed_trajectory(trajectories)
    hash_map = {}
    for trajectory in trajectories:
        if trajectory.refe_traj_id == -1:
            hash_map[trajectory.traj_id] = trajectory
    total_ped = 0
    for trajectory in trajectories:
        ped = restore_trajectory_ped_error(trajectory, hash_map)
        total_ped += ped
    print("ped error:", total_ped / len(trajectories))
    print("max ped error:", max_ped_error)
