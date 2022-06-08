import time

import pandas as pd

from Experiment.DP.dp import douglas_peucker, td_tr
from Experiment.TrajSTCCluster.stcCluster.cluster import cluster_HAC
from Experiment.common.zip import zip_compress
from Experiment.compare import compare
from Experiment.data.data_process import get_trajectories, get_berlin_mod_0_005_trajectories

if __name__ == '__main__':
    epsilon = 3.0
    trajectories = get_berlin_mod_0_005_trajectories("point_list")
    dp_start_time = time.perf_counter()
    dp_trajectories = []
    # dp 单轨迹压缩
    for trajectory in trajectories:
        sample_index = douglas_peucker(trajectory, start=0, last=len(trajectory) - 1, epsilon=epsilon / 100000.0)
        dp_trajectory = []
        for e in sample_index:
            dp_trajectory.append(trajectory[e])
        dp_trajectories.append(dp_trajectory)

    dp_end_time = time.perf_counter()

    cluster_start_time = time.perf_counter()
    # 聚类多轨迹压缩
    cluster_HAC(dp_trajectories, t=epsilon / 100000.0)

    cluster_end_time = time.perf_counter()

    # 输出 mtc 之后的数据
    compressed_trajectories = []
    for trajectory in dp_trajectories:
        for ele in trajectory:
            compressed_trajectories.append([int(ele.t), round(ele.x, 4), round(ele.y, 4)])
    pd.DataFrame(compressed_trajectories).to_csv("mtc_dpCluster_test.txt", index=False, header=0)

    # 分析误差
    res = []
    origin_trajectories = get_berlin_mod_0_005_trajectories(trajectory_type="point_list")
    average_ped_error = 0
    max_ped_error = 0
    average_sed_error = 0
    max_sed_error = 0
    average_speed_error = 0
    max_speed_error = 0
    average_angle_error = 0
    max_angle_error = 0

    for i in range(len(dp_trajectories)):
        [a, b] = compare.get_PED_error(origin_trajectories[i], dp_trajectories[i])
        [c, d] = compare.get_SED_error(origin_trajectories[i], dp_trajectories[i])
        [e, f] = compare.get_speed_error(origin_trajectories[i], dp_trajectories[i])
        [g, h] = compare.get_angle_error(origin_trajectories[i], dp_trajectories[i])
        average_ped_error += a
        max_ped_error = max(max_ped_error, b)
        average_sed_error += c
        max_sed_error = max(max_sed_error, d)
        average_speed_error += e
        max_speed_error = max(max_speed_error, f)
        average_angle_error += g
        max_angle_error = max(max_angle_error, h)
    print("average_ped_error:", average_ped_error / len(origin_trajectories))
    print("max_ped_error:", max_ped_error)
    print("average_sed_error:", average_sed_error / len(origin_trajectories))
    print("max_sed_error:", max_sed_error)
    print("average_speed_error:", average_speed_error / len(origin_trajectories))
    print("max_speed_error:", max_speed_error)
    print("average_angle_error:", average_angle_error / len(origin_trajectories))
    print("max_angle_error:", max_angle_error)
    [a, b] = zip_compress("mtc_dpCluster_test.txt")
    print(["epsilon", "average_ped_error", "max_ped_error", "average_sed_error", "max_sed_error", "average_speed_error",
           "max_speed_error", "average_angle_error", "max_angle_error", "before_zip_size", "after_zip_size", "dp_time",
           "cluster_time"])
    res.append(
        [epsilon, average_ped_error / len(trajectories), max_ped_error, average_sed_error / len(trajectories),
         max_sed_error, average_speed_error / len(trajectories), max_speed_error,
         average_angle_error / len(trajectories), max_angle_error, a, b, dp_end_time - dp_start_time,
         cluster_end_time - cluster_start_time])
    print(res)
