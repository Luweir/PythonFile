import numpy as np
import pandas as pd
import math
import random

from typing import List

from Experiment.Similarity_based_MTC.similarity_based.mtc import mtc_add
from Experiment.common.Point import Point
from Experiment.common.Trajectory import Trajectory
from Experiment.common.zip import zip_compress
from Experiment.compare.compare import get_PED_error
from Experiment.data.DataProcess.data_process import get_trajectories


def output_compressed_trajectory(trajectories: List[Trajectory]):
    """
    输出压缩轨迹
    :param trajectories: 压缩后的轨迹集
    :return:
    """
    data = []
    all_len = 0
    for trajectory in trajectories:
        all_len += len(trajectory.points)
        for point in trajectory.points:
            if point.p is None:
                data.append([int(point.t), round(point.x, 4), round(point.y, 4)])
            else:
                data.append([int(point.t), int(point.p.t)])
    pd.DataFrame(data).to_csv("mtc_similarity_compressed_trajectory.txt", header=0,
                              index=False)
    print("压缩后总点数：", all_len)


# ped 误差
def ped_error(trajectories: List[Trajectory], compressed_trajectories: List[Trajectory]) -> list:
    """
    先将压缩轨迹恢复，然后与原始轨迹比较 获得PED误差
    :param trajectories: 原始轨迹
    :param compressed_trajectories: 压缩轨迹
    :return: [平均误差，最大误差]
    """
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
    # epsilon = 50
    res = []
    for i in range(1, 40):
        epsilon = 25 * i
        trajectories = get_trajectories()
        compressed_trajectories = []
        for trajectory in trajectories:
            mtc_add(trajectory, compressed_trajectories, epsilon)
        output_compressed_trajectory(compressed_trajectories)
        trajectories = get_trajectories()
        [average_ped_error, max_ped_error] = ped_error(trajectories, compressed_trajectories)
        print("平均 ped 误差：", average_ped_error)
        print("最大 ped 误差：", max_ped_error)
        [a, b] = zip_compress("mtc_similarity_compressed_trajectory.txt")
        res.append([epsilon, average_ped_error, max_ped_error, a, b])
    res = pd.DataFrame(res, columns=['误差阈值', '平均ped误差', '最大ped误差', '压缩后文件大小', 'zip后文件大小'])
