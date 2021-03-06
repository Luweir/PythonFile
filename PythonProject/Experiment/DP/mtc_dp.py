import time

import pandas as pd

from Experiment.DP.dp import douglas_peucker, new_dp
from Experiment.common.zip import zip_compress, zip_decompress
from Experiment.compare.compare import get_PED_error, get_SED_error, get_speed_error, get_angle_error, get_dtw
from Experiment.data.data_process import get_trajectories, get_berlin_mod_0_005_trajectories, get_walk_data, \
    get_airline_data


def run():
    # epsilon = 50
    res = []
    for i in range(1, 100):
        epsilon = 5*i
        # 加载数据
        trajectories = get_berlin_mod_0_005_trajectories(trajectory_type="point_list")
        compressed_trajectories = []
        data = []
        average_ped_error = 0
        max_ped_error = 0
        average_sed_error = 0
        max_sed_error = 0
        average_speed_error = 0
        max_speed_error = 0
        average_angle_error = 0
        max_angle_error = 0
        compress_start_time = 0
        compress_end_time = 0
        dtw_distance = 0

        for trajectory in trajectories:
            compress_start_time = time.perf_counter()
            sample_index = douglas_peucker(trajectory, 0, len(trajectory) - 1, epsilon / 100000.0)
            compress_end_time = time.perf_counter()

            compressed_trajectory = []
            for index in sample_index:
                compressed_trajectory.append(trajectory[index])
                data.append(trajectory[index].to_list())

            [a, b] = get_PED_error(trajectory, compressed_trajectory)
            [c, d] = get_SED_error(trajectory, compressed_trajectory)
            [e, f] = get_speed_error(trajectory, compressed_trajectory)
            [g, h] = get_angle_error(trajectory, compressed_trajectory)

            average_ped_error += a
            max_ped_error = max(max_ped_error, b)
            average_sed_error += c
            max_sed_error = max(max_sed_error, d)
            average_speed_error += e
            max_speed_error = max(max_speed_error, f)
            average_angle_error += g
            max_angle_error = max(max_angle_error, h)
            dtw_distance += get_dtw(trajectory, compressed_trajectory)
            compressed_trajectories += compressed_trajectory

        pd.DataFrame(data).to_csv("mtc_dp_compressed_trajectory.txt", header=0, index=False)
        [a, b] = zip_compress("mtc_dp_compressed_trajectory.txt")
        decompress_time = zip_decompress("mtc_dp_compressed_trajectory.zip")

        average_ped_error /= len(trajectories)
        average_sed_error /= len(trajectories)
        average_speed_error /= len(trajectories)
        average_angle_error /= len(trajectories)
        dtw_distance /= len(trajectories)
        restore_time = 0

        print("距离阈值:", epsilon)
        print("average_ped_error:", average_ped_error)
        print("max_ped_error:", max_ped_error)
        print("average_sed_error:", average_sed_error)
        print("max_sed_error:", max_sed_error)
        print("average_speed_error:", average_speed_error)
        print("max_speed_error:", max_speed_error)
        print("average_angle_error:", average_angle_error)
        print("max_speed_error:", max_angle_error)
        print("average_dtw_distance:", dtw_distance)
        print("decompress_time:", decompress_time)
        print("restore_time:", restore_time)
        print("压缩时间：", (compress_end_time - compress_start_time))
        print("点数：", len(compressed_trajectories))

        res.append(
            [epsilon, average_ped_error, max_ped_error, average_sed_error, max_sed_error,
             average_speed_error, max_speed_error, average_angle_error, max_angle_error, dtw_distance,
             a, b, (compress_end_time - compress_start_time), decompress_time, restore_time])

    res = pd.DataFrame(res, columns=['误差阈值', '平均ped误差', '最大ped误差', '平均sed误差', '最大sed误差', '平均速度误差', '最大速度误差', '平均角度误差',
                                     '最大角度误差', '平均dtw距离', '压缩后文件大小', 'zip后文件大小', '压缩时间(s)', '解压时间(s)', "恢复时间(s)"])
    return res


if __name__ == '__main__':
    res = run()
