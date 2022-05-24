import pandas as pd

from Experiment.DP.dp import douglas_peucker
from Experiment.common.zip import zip_compress
from Experiment.compare.compare import get_PED_error
from Experiment.data.DataProcess.data_process import get_trajectories

if __name__ == '__main__':
    epsilon = 200
    trajectories = get_trajectories(trajectory_type="point_list")
    compressed_trajectories = []
    data = []
    average_ped_error = 0
    max_ped_error = 0
    for trajectory in trajectories:
        sample_index = douglas_peucker(trajectory, 0, len(trajectory) - 1, epsilon / 100000.0)
        compressed_trajectory = []
        for index in sample_index:
            compressed_trajectory.append(trajectory[index])
            data.append(trajectory[index].to_list())
        [a, b] = get_PED_error(trajectory, compressed_trajectory)
        average_ped_error += a
        max_ped_error = max(max_ped_error, b)
        compressed_trajectories += compressed_trajectory
    pd.DataFrame(data).to_csv("mtc_dp_compressed_trajectory.txt", header=0, index=False)
    print("average_ped_error:", average_ped_error / len(trajectories))
    print("max_ped_error:", max_ped_error)
    zip_compress("mtc_dp_compressed_trajectory.txt")
