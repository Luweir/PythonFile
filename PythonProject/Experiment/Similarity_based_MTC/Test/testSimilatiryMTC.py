import numpy as np
import pandas as pd
import math
import random

from Experiment.Similarity_based_MTC.similarity_based.Point import Point
from Experiment.Similarity_based_MTC.similarity_based.Trajectory import Trajectory
from Experiment.Similarity_based_MTC.similarity_based.mtc import mtc, mtc_add
from Experiment.compare_result.compare import get_PED_error


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


def generate_origin_trajectory(data):
    trajectory = []
    for i in range(len(data)):
        trajectory.append([data.iloc[i][0], data.iloc[i][1], data.iloc[i][2]])
    return trajectory


def output_compressed_trajectory(trajectories):
    data = []
    for trajectory in trajectories:
        for point in trajectory.points:
            if point.t2 is None:
                data.append([point.t, point.x, point.y])
            else:
                data.append([point.t, int(point.t2.t)])
    pd.DataFrame(data).to_csv("output_compressed_trajectory.csv", header=False,
                              index=False)


# ped 误差
def ped_error(trajectories: list, compressed_trajectories: list):
    restore_trajectories = []
    for trajectory in compressed_trajectories:
        restore_single_trajectory = []
        for i in range(len(trajectory.points)):
            cur_point = trajectory.points[i]
            while cur_point.t2 is not None:
                cur_point = cur_point.t2
            restore_single_trajectory.append([trajectory.points[i].t, cur_point.x, cur_point.y])
        restore_trajectories.append(restore_single_trajectory)
    max_ped_error = 0
    ped_error = 0
    for i in range(len(restore_trajectories)):
        [a, b] = get_PED_error(trajectories[i], restore_trajectories[i])
        ped_error += a
        max_ped_error = max(max_ped_error, b)
    return [ped_error / len(restore_trajectories), max_ped_error]


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
                    Trajectory(6, generate_trajectory(pd.read_csv("output_origin_trajectory_6.csv", header=None)))
                    ]
    # trajectories = [Trajectory(0, generate_trajectory(data1)),
    #                 Trajectory(1, generate_trajectory(data1)),
    #                 Trajectory(2, generate_trajectory(data1)),
    #                 Trajectory(3, generate_trajectory(data1)),
    #                 Trajectory(4, generate_trajectory(data1)),
    #                 Trajectory(5, generate_trajectory(data1)),
    #                 Trajectory(6, generate_trajectory(data1))]
    compressed_trajectories = []
    for trajectory in trajectories:
        mtc_add(trajectory, compressed_trajectories, 1500)
    output_compressed_trajectory(compressed_trajectories)
    trajectories = [generate_origin_trajectory(pd.read_csv("output_origin_trajectory_0.csv", header=None)),
                    generate_origin_trajectory(pd.read_csv("output_origin_trajectory_1.csv", header=None)),
                    generate_origin_trajectory(pd.read_csv("output_origin_trajectory_2.csv", header=None)),
                    generate_origin_trajectory(pd.read_csv("output_origin_trajectory_3.csv", header=None)),
                    generate_origin_trajectory(pd.read_csv("output_origin_trajectory_4.csv", header=None)),
                    generate_origin_trajectory(pd.read_csv("output_origin_trajectory_5.csv", header=None)),
                    generate_origin_trajectory(pd.read_csv("output_origin_trajectory_6.csv", header=None))
                    ]
    [ped_error, max_ped_error] = ped_error(trajectories, compressed_trajectories)
    print("平均 ped 误差：", ped_error)
    print("最大 ped 误差：", max_ped_error)
