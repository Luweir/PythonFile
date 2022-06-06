import random
import sys

import numpy as np
import pandas as pd

from Experiment.common.Point import Point


def output_data(trajectories):
    name = 1
    for trajectory in trajectories:
        pd.DataFrame(trajectory).to_csv("select_trajectory_" + str(name) + ".csv", index=False, header=0)
        name += 1


if __name__ == '__main__':
    data = pd.read_excel("./trips.xlsx").values.tolist()
    data_len = len(data)
    true_traj_count = 1
    try_count = 0
    a = random.randint(0, data_len - 1000)
    # 选定一条轨迹
    trajectories = [data[a:a + 1000]]
    # 看能不能找出另外相似的9条轨迹  他们互相相似
    while true_traj_count < 10:
        b = random.randint(0, data_len - 1000)
        try_count += 1
        trajectory_b = data[b:b + 1000]
        flag = True
        for trajectory_a in trajectories:
            if abs(b - trajectory_a[0][0]) < 1000:
                flag = False
                break
            sum_ped_error = 0
            # max_ped_error = sys.maxsize
            for i in range(1000):
                ped_error = Point(x=trajectory_a[i][2], y=trajectory_a[i][3]).distance(
                    Point(x=trajectory_b[i][2], y=trajectory_b[i][3]))
                sum_ped_error += ped_error
                # max_ped_error = max(max_ped_error, ped_error)
            print(sum_ped_error / 1000)
            if sum_ped_error / 1000 > 0.02:
                flag = False
                break
        # 如果这个轨迹跟集合里面的轨迹都很接近 接受
        if flag:
            trajectories.append(trajectory_b)
            true_traj_count += 1
            print("------------------true_traj_count:", true_traj_count, "----------------------")
        # 如果尝试次数超过1000次 说明随机选的a有问题 重新选
        if try_count >= 1000 and true_traj_count <= 5:
            true_traj_count = 1
            try_count = 0
            a = random.randint(0, data_len - 1000)
            trajectories = [data[a:a + 1000]]
