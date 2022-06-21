import copy
import time

import pandas as pd

from typing import List

from Experiment.TrajStore.src.cluster import traj_store_cluster
from Experiment.common.Point import Point

from Experiment.common.Trajectory import Trajectory
from Experiment.common.zip import zip_compress
from Experiment.compare.compare import get_PED_error, get_SED_error, get_speed_error, get_angle_error
from Experiment.data.data_process import get_trajectories, get_berlin_mod_0_005_trajectories


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
        # 有参考轨迹 就只用存映射时间
        else:
            data += [[ele] for ele in trajectory.reference_time]
    pd.DataFrame(data).to_csv("output_compressed_trajectory.txt", header=False,
                              index=False)


def get_restore_trajectory(compressed_trajectory: Trajectory, hash_map):
    """
    恢复已压缩轨迹
    :param compressed_trajectory: 压缩后的轨迹
    :param hash_map: 参考轨迹映射 <trajectory_id,trajectory>
    :return: 恢复后的轨迹
    """
    # 无参考轨迹
    if compressed_trajectory.reference_trajectory_id == -1:
        return compressed_trajectory
    # 有参考轨迹 则进行轨迹恢复   先把时间映射的点给恢复
    reference_trajectory = hash_map[compressed_trajectory.reference_trajectory_id]
    restore_points = []
    for i in range(len(compressed_trajectory.reference_time)):
        restore_points.append(Point(x=reference_trajectory.points[i].x, y=reference_trajectory.points[i].y,
                                    t=int(compressed_trajectory.reference_time[i])))
    return Trajectory(compressed_trajectory.trajectory_id, restore_points)


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
        new_trajectory_points.append(trajectory.points[0])
        first_index = 0
        last_index = first_index + 1
        while last_index < len(trajectory.points):
            first_point = trajectory.points[first_index]
            last_point = trajectory.points[last_index]
            flag = True
            for mid_index in range(first_index + 1, last_index):
                mid_point = trajectory.points[mid_index]
                temp_point = mid_point.linear_prediction(first_point, last_point)
                if mid_point.get_haversine(temp_point) > epsilon:
                    flag = False
                    break
            if not flag or last_index == len(trajectory.points) - 1:
                new_trajectory_points.append(trajectory.points[last_index - 1])
                first_index = last_index - 1
            last_index += 1
        # 加入最后一个点
        new_trajectory_points.append(trajectory.points[-1])
        linear_eliminate_trajectories.append(Trajectory(trajectory.trajectory_id, new_trajectory_points))
    return linear_eliminate_trajectories


# 测试多组epsilon 压缩
def run():
    res = []
    for i in range(1, 150):
        average_ped_error = 0
        max_ped_error = 0
        average_sed_error = 0
        max_sed_error = 0
        average_speed_error = 0
        max_speed_error = 0
        average_angle_error = 0
        max_angle_error = 0
        epsilon = 20 * i

        # 测试柏林数据集
        trajectories = get_trajectories()
        compress_start_time = time.perf_counter()

        # 第一部分 线性法消除无关点   ！！！实验过程中发现 这个线性消除的阈值不能过大 否则多轨迹聚类效果很差
        linear_eliminate_trajectories = linear_eliminate(trajectories, 0.2 * epsilon)
        # linear_eliminate_trajectories = copy.deepcopy(trajectories)

        # 第二部分 聚类压缩  原理：利用轨迹经过的路径相近而只存非参考轨迹的映射时间
        group = traj_store_cluster(linear_eliminate_trajectories, 0.4 * epsilon)
        compress_end_time = time.perf_counter()
        output_compressed_trajectory(linear_eliminate_trajectories)

        # 第三部分 轨迹恢复和误差测量
        hash_map = {}
        for trajectory in linear_eliminate_trajectories:
            if trajectory.reference_trajectory_id == -1:
                hash_map[trajectory.trajectory_id] = trajectory
        for i in range(len(linear_eliminate_trajectories)):
            restore_trajectory = get_restore_trajectory(linear_eliminate_trajectories[i], hash_map)
            [a, b] = get_PED_error(trajectories[i].points, restore_trajectory.points)
            [c, d] = get_SED_error(trajectories[i].points, restore_trajectory.points)
            [e, f] = get_speed_error(trajectories[i].points, restore_trajectory.points)
            [g, h] = get_angle_error(trajectories[i].points, restore_trajectory.points)
            average_ped_error += a
            max_ped_error = max(max_ped_error, b)
            average_sed_error += c
            max_sed_error = max(max_sed_error, d)
            average_speed_error += e
            max_speed_error = max(max_speed_error, f)
            average_angle_error += g
            max_angle_error = max(max_angle_error, h)

        print("average_ped_error:", average_ped_error / len(trajectories))
        print("max_ped_error:", max_ped_error)
        print("average_sed_error:", average_sed_error / len(trajectories))
        print("max_sed_error:", max_sed_error)
        print("average_speed_error:", average_speed_error / len(trajectories))
        print("max_speed_error:", max_speed_error)
        print("average_angle_error:", average_angle_error / len(trajectories))
        print("max_speed_error:", max_angle_error)
        [a, b] = zip_compress("output_compressed_trajectory.txt")
        res.append(
            [epsilon, average_ped_error / len(trajectories), max_ped_error, average_sed_error / len(trajectories),
             max_sed_error, average_speed_error / len(trajectories), max_speed_error,
             average_angle_error / len(trajectories), max_angle_error, a, b, (compress_end_time - compress_start_time)])
    res = pd.DataFrame(res, columns=['误差阈值', '平均ped误差', '最大ped误差', '平均sed误差', '最大sed误差', '平均速度误差', '最大速度误差', '平均角度误差',
                                     '最大角度误差', '压缩后文件大小', 'zip后文件大小', '压缩时间(s)'])
    return res


# 测试单个epsilon 压缩
def run_sample():
    epsilon = 500
    res = []
    average_ped_error = 0
    max_ped_error = 0
    average_sed_error = 0
    max_sed_error = 0
    average_speed_error = 0
    max_speed_error = 0
    average_angle_error = 0
    max_angle_error = 0

    # 测试柏林数据集
    trajectories = get_trajectories()
    compress_start_time = time.perf_counter()

    # 第一部分 线性法消除无关点   ！！！实验过程中发现 这个线性消除的阈值不能过大 否则多轨迹聚类效果很差
    linear_eliminate_trajectories = linear_eliminate(trajectories, 0.2 * epsilon)
    # linear_eliminate_trajectories = copy.deepcopy(trajectories)

    # 第二部分 聚类压缩  原理：利用轨迹经过的路径相近而只存非参考轨迹的映射时间
    group = traj_store_cluster(linear_eliminate_trajectories, 0.4 * epsilon)
    compress_end_time = time.perf_counter()
    output_compressed_trajectory(linear_eliminate_trajectories)

    # 第三部分 轨迹恢复和误差测量
    hash_map = {}
    for trajectory in linear_eliminate_trajectories:
        if trajectory.reference_trajectory_id == -1:
            hash_map[trajectory.trajectory_id] = trajectory
    for i in range(len(linear_eliminate_trajectories)):
        restore_trajectory = get_restore_trajectory(linear_eliminate_trajectories[i], hash_map)
        [a, b] = get_PED_error(trajectories[i].points, restore_trajectory.points)
        [c, d] = get_SED_error(trajectories[i].points, restore_trajectory.points)
        [e, f] = get_speed_error(trajectories[i].points, restore_trajectory.points)
        [g, h] = get_angle_error(trajectories[i].points, restore_trajectory.points)
        average_ped_error += a
        max_ped_error = max(max_ped_error, b)
        average_sed_error += c
        max_sed_error = max(max_sed_error, d)
        average_speed_error += e
        max_speed_error = max(max_speed_error, f)
        average_angle_error += g
        max_angle_error = max(max_angle_error, h)
        # 加个轨迹的相似度比较

    print("average_ped_error:", average_ped_error / len(trajectories))
    print("max_ped_error:", max_ped_error)
    print("average_sed_error:", average_sed_error / len(trajectories))
    print("max_sed_error:", max_sed_error)
    print("average_speed_error:", average_speed_error / len(trajectories))
    print("max_speed_error:", max_speed_error)
    print("average_angle_error:", average_angle_error / len(trajectories))
    print("max_speed_error:", max_angle_error)
    [a, b] = zip_compress("output_compressed_trajectory.txt")
    res.append(
        [epsilon, average_ped_error / len(trajectories), max_ped_error, average_sed_error / len(trajectories),
         max_sed_error, average_speed_error / len(trajectories), max_speed_error,
         average_angle_error / len(trajectories), max_angle_error, a, b, (compress_end_time - compress_start_time)])
    res = pd.DataFrame(res, columns=['误差阈值', '平均ped误差', '最大ped误差', '平均sed误差', '最大sed误差', '平均速度误差', '最大速度误差', '平均角度误差',
                                     '最大角度误差', '压缩后文件大小', 'zip后文件大小', '压缩时间(s)'])
    return res


if __name__ == '__main__':
    res = run_sample()
    # res = run_sample()
