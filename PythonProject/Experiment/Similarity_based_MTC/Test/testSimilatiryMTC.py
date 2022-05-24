import numpy as np
import pandas as pd
import math
import random

from typing import List

from Experiment.Similarity_based_MTC.similarity_based.mtc import mtc_add
from Experiment.common.Point import Point
from Experiment.common.Trajectory import Trajectory
from Experiment.compare.compare import get_PED_error
from Experiment.data.DataProcess.data_process import get_trajectories


def output_compressed_trajectory(trajectories):
    data = []
    all_len = 0
    for trajectory in trajectories:
        all_len += len(trajectory.points)
        for point in trajectory.points:
            if point.p is None:
                data.append([int(point.t), round(point.x, 4), round(point.y, 4)])
            else:
                data.append([int(point.t), int(point.p.t)])
    pd.DataFrame(data).to_csv("output_compressed_trajectory.txt", header=0,
                              index=False)
    print("压缩后总点数：", all_len)


# ped 误差
def ped_error(trajectories: List[Trajectory], compressed_trajectories: List[Trajectory]):
    restore_trajectories = []
    for trajectory in compressed_trajectories:
        restore_single_trajectory = []
        for i in range(len(trajectory.points)):
            cur_point = trajectory.points[i]
            while cur_point.p is not None:
                cur_point = cur_point.p
            restore_single_trajectory.append(Point(cur_point.x, cur_point.y, t=trajectory.points[i].t))
        restore_trajectories.append(restore_single_trajectory)
    max_ped_error = 0
    ped_error = 0
    for i in range(len(restore_trajectories)):
        [a, b] = get_PED_error(trajectories[i].points, restore_trajectories[i])
        ped_error += a
        max_ped_error = max(max_ped_error, b)
    return [ped_error / len(restore_trajectories), max_ped_error]


if __name__ == '__main__':
    trajectories = get_trajectories()
    compressed_trajectories = []
    for trajectory in trajectories:
        mtc_add(trajectory, compressed_trajectories, 200)
    output_compressed_trajectory(compressed_trajectories)
    trajectories = get_trajectories()
    [ped_error, max_ped_error] = ped_error(trajectories, compressed_trajectories)
    print("平均 ped 误差：", ped_error)
    print("最大 ped 误差：", max_ped_error)
