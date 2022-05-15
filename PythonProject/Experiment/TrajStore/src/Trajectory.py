import numpy as np
import pandas as pd


# ---------------------------------------------
# author: luweir
# target: trajectory class
# date: 2022-5-13
# ---------------------------------------------

class Trajectory:
    traj_id = -1  # 轨迹自身ID
    points = []  # 自身轨迹点
    traj_size = 0  # 轨迹点长度
    refe_traj_id = -1  # 参考轨迹ID
    refe_time = []  # 对应在参考轨迹上的时间

    def __init__(self, traj_id, trajectory=None):
        self.traj_id = traj_id
        self.refe_traj_id = -1
        self.refe_time = []
        self.points = []
        if trajectory:
            self.points = trajectory
            self.traj_size = len(trajectory)

    def add_point(self, point):
        self.points.append(point)

    def to_list(self):
        trajectory_list = []
        for ele in self.points:
            trajectory_list.append([ele.t, ele.x, ele.y])
        return trajectory_list
