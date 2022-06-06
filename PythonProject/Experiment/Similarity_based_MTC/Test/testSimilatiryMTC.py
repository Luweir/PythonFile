import time

import pandas as pd

from typing import List

from Experiment.Similarity_based_MTC.similarity_based.mtc import mtc_add
from Experiment.common.Point import Point
from Experiment.common.Trajectory import Trajectory
from Experiment.common.zip import zip_compress
from Experiment.compare.compare import get_PED_error, get_SED_error, get_speed_error, get_angle_error
from Experiment.data.data_process import get_trajectories


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
def get_error(trajectories: List[Trajectory], compressed_trajectories: List[Trajectory]) -> list:
    """
    先将压缩轨迹恢复，然后与原始轨迹比较 获得PED误差
    :param trajectories: 原始轨迹
    :param compressed_trajectories: 压缩轨迹
    :return: [平均误差，最大误差]
    """
    average_ped_error = 0
    max_ped_error = 0
    average_sed_error = 0
    max_sed_error = 0
    average_speed_error = 0
    max_speed_error = 0
    average_angle_error = 0
    max_angle_error = 0
    restore_trajectories = []
    for trajectory in compressed_trajectories:
        restore_single_trajectory = []
        for i in range(len(trajectory.points)):
            cur_point = trajectory.points[i]
            while cur_point.p is not None:
                cur_point = cur_point.p
            restore_single_trajectory.append(Point(cur_point.x, cur_point.y, t=trajectory.points[i].t))
        restore_trajectories.append(restore_single_trajectory)
    for i in range(len(restore_trajectories)):
        [a, b] = get_PED_error(trajectories[i].points, restore_trajectories[i])
        [c, d] = get_SED_error(trajectories[i].points, restore_trajectories[i])
        [e, f] = get_speed_error(trajectories[i].points, restore_trajectories[i])
        [g, h] = get_angle_error(trajectories[i].points, restore_trajectories[i])
        average_ped_error += a
        max_ped_error = max(max_ped_error, b)
        average_sed_error += c
        max_sed_error = max(max_sed_error, d)
        average_speed_error += e
        max_speed_error = max(max_speed_error, f)
        average_angle_error += g
        max_angle_error = max(max_angle_error, h)
    return [average_ped_error / len(trajectories), max_ped_error, average_sed_error / len(trajectories), max_sed_error,
            average_speed_error / len(trajectories), max_speed_error, average_angle_error / len(trajectories),
            max_angle_error]


def run():
    # epsilon = 50
    res = []
    for i in range(1, 40):
        epsilon = 25 * i
        trajectories = get_trajectories()
        compressed_trajectories = []
        compress_start_time = time.perf_counter()
        for trajectory in trajectories:
            mtc_add(trajectory, compressed_trajectories, epsilon)
        compress_end_time = time.perf_counter()
        output_compressed_trajectory(compressed_trajectories)
        trajectories = get_trajectories()
        [a, b, c, d, e, f, g, h] = get_error(trajectories, compressed_trajectories)
        print("average_ped_error:", a / len(trajectories))
        print("max_ped_error:", b)
        print("average_sed_error:", c / len(trajectories))
        print("max_sed_error:", d)
        print("average_speed_error:", e / len(trajectories))
        print("max_speed_error:", f)
        print("average_angle_error:", g / len(trajectories))
        print("max_speed_error:", h)
        [m, n] = zip_compress("mtc_similarity_compressed_trajectory.txt")
        res.append([epsilon, a, b, c, d, e, f, g, h, m, n, (compress_end_time - compress_start_time)])
    res = pd.DataFrame(res, columns=['误差阈值', '平均ped误差', '最大ped误差', '平均sed误差', '最大sed误差', '平均速度误差', '最大速度误差', '平均角度误差',
                                     '最大角度误差', '压缩后文件大小', 'zip后文件大小', '压缩时间(s)'])
    return res


if __name__ == '__main__':
    res = run()
