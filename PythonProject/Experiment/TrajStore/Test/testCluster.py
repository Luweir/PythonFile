import numpy as np
import pandas as pd
import math
from Experiment.TrajStore.src.cluster import trajStore_cluster
from Experiment.TrajStore.src.Point import Point
from Experiment.TrajStore.src.Trajectory import Trajectory


def generate_trajectory(data):
    trajectory = []
    for i in range(len(data)):
        trajectory.append(Point(data.iloc[i][1], data.iloc[i][2], data.iloc[i][0]))
    return trajectory


if __name__ == '__main__':
    data1 = pd.read_csv("../../data/10.9.csv", header=None)
    data2 = pd.read_csv("../../data/10.9.csv", header=None)
    trajectories = [Trajectory(1, generate_trajectory(data1)), Trajectory(2, generate_trajectory(data2))]
    print(trajectories)
    group = trajStore_cluster(trajectories, 10000)
    print(group)
