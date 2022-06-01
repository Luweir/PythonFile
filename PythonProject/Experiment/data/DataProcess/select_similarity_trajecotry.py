import random

import numpy as np
import pandas as pd

from Experiment.common.Point import Point

if __name__ == '__main__':
    data = pd.read_excel("./trips.xlsx").values.tolist()
    data_len = len(data)
    a = 0
    trajectories = []
    while a + 1000 < data_len:
        true_traj_count = 1
        # 选定一条轨迹
        trajectories = [data[a:a + 1000]]
        b = a + 1000
        # 看能不能找出另外相似的9条轨迹  他们互相相似
        while b + 1000 < data_len and true_traj_count < 10:
            # b = random.randint(0, data_len - 1000)
            trajectory_b = data[b:b + 1000]
            flag = True
            for trajectory_a in trajectories:
                sum_ped_error = 0
                for i in range(1000):
                    sum_ped_error += Point(x=trajectory_a[i][1], y=trajectory_a[i][2]).distance(
                        Point(x=trajectory_b[i][1], y=trajectory_b[i][2]))
                print(sum_ped_error)
                if sum_ped_error > 30:
                    flag = False
                    break
            if flag:
                trajectories.append(trajectory_b)
                true_traj_count += 1
                b += 1000
            else:
                b += 50
        if true_traj_count == 10:
            break
        a += 50
