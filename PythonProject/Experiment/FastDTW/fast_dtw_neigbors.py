import numpy as np
import pandas as pd
from fastdtw import fastdtw
import math

EARTH_RADIUS = 6371229  # m 用于两点间距离计算


def get_haversine(point_a, point_b):
    lat1 = point_a[1] * math.pi / 180
    lat2 = point_b[1] * math.pi / 180
    lon1 = point_a[2] * math.pi / 180
    lon2 = point_b[2] * math.pi / 180
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a_a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a_a), math.sqrt(1 - a_a))
    return EARTH_RADIUS * c


def euc_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_distance(p1, p2):
    return abs(p1[1] - p2[1])


def get_fastdtw(traj1, traj2):
    distance, path = fastdtw(traj1, traj2, dist=get_haversine)
    return distance
    # print("" + str(distance / 1000000))  # km


if __name__ == '__main__':
    traj1 = [[0, 1.1, 2.1], [2, 1.5, 2.1]]
    traj2 = [[0, 1.1, 2.1], [1, 1.3, 2.1], [2, 1.5, 2.1]]
    sum_distance = get_haversine(traj2[1], traj2[0])
    print(sum_distance)
    dtw = get_fastdtw(np.array(traj1), np.array(traj2))
    print(dtw)
    print(1 - dtw / sum_distance)
