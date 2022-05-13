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
        # trajectory.append(Point(data.iloc[i][1] + random.uniform(-0.01, 0.01),
        #                         data.iloc[i][2] + random.uniform(-0.01, 0.01),
        #                         data.iloc[i][0] + random.randint(-5, 5)))
        trajectory.append(Point(data.iloc[i][1],
                                data.iloc[i][2],
                                data.iloc[i][0]))
    return trajectory


def output_origin_trajectory(trajectory):
    data = []
    start_time = trajectory.points[0].t
    for point in trajectory.points:
        data.append([point.t - start_time, round(point.x, 4), round(point.y, 4)])
    pd.DataFrame(data).to_csv("output_origin_trajectory_" + str(trajectory.traj_id) + ".csv", header=False, index=False)


def output_compressed_trajectory(trajectory):
    data = []
    start_time = trajectory.points[0].t
    # 自身就是本聚类里面的参考轨迹
    if trajectory.refe_traj_id == -1:
        for point in trajectory.points:
            data.append([point.t - start_time, round(point.x, 4), round(point.y, 4)])
        pd.DataFrame(data).to_csv("output_compressed_trajectory_" + str(trajectory.traj_id) + ".csv", header=False,
                                  index=False)
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
        pd.DataFrame(data).to_csv("output_compressed_trajectory_" + str(trajectory.traj_id) + ".csv", header=False,
                                  index=False)


if __name__ == '__main__':
    # data1 = pd.read_csv("../../data/10.9.csv", header=None)
    # data2 = pd.read_csv("../../data/10.9.csv", header=None)
    # data3 = pd.read_csv("../../data/10.11.csv", header=None)
    trajectories = [Trajectory(0, generate_trajectory(pd.read_csv("output_origin_trajectory_0.csv", header=None))),
                    Trajectory(1, generate_trajectory(pd.read_csv("output_origin_trajectory_1.csv", header=None))),
                    Trajectory(2, generate_trajectory(pd.read_csv("output_origin_trajectory_2.csv", header=None))),
                    Trajectory(3, generate_trajectory(pd.read_csv("output_origin_trajectory_3.csv", header=None))),
                    Trajectory(4, generate_trajectory(pd.read_csv("output_origin_trajectory_4.csv", header=None))),
                    Trajectory(5, generate_trajectory(pd.read_csv("output_origin_trajectory_5.csv", header=None))),
                    Trajectory(6, generate_trajectory(pd.read_csv("output_origin_trajectory_6.csv", header=None)))]
    print(trajectories)
    group = trajStore_cluster(trajectories,25000)
    print(group)
