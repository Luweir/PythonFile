# -*- coding: utf-8 -*-
import math
import numpy as np
import sys
import traj_dist.distance as tdist
import pickle

import importlib


def test_dist():
    with open("E:/Desktop/traj-dist-master/data/benchmark_trajectories.pkl", 'rb') as f:
        content = pickle.load(f, encoding='iso-8859-1')
    print(content)
    traj_list = content[
                :10]
    traj_A = traj_list[0]
    traj_B = traj_list[1]
    print("trj_A", traj_A)
    # Simple distance

    dist = tdist.sspd(traj_A, traj_B)
    print(dist)

    # Pairwise distance

    pdist = tdist.pdist(traj_list, metric="sspd")
    print(pdist)

    # Distance between two list of trajectories

    cdist = tdist.cdist(traj_list, traj_list, metric="sspd")
    print(cdist)


if __name__ == '__main__':
    # print(np.arange(1, 8, 1))
    # print(np.linspace(30, 41, 8))
    # list1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # list1 = np.array(list1)
    # print(sys.getsizeof(1000000000000000000000))
    # s = '00011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001'
    # b = int(s, 2)
    # print(b)
    # print(format(b, 'b'))
    # print(1 << (5 * 8 + 8 - 1 - 0))
    # print("-------------------------------------")
    # map = {}
    # map[1] = 123123123123
    # map[2] = 123123
    # print(map.keys())
    # print(map.values())
    # print(map[1] == 123123123123)
    print("-------------------")
    # importlib.reload(sys)
    # sys.getfilesystemencoding()
    # test_dist()
    dic = {}
    for i in range(30):
        dic[i] = i
    # dic[1] = 1
    # dic[2] = 2
    # dic[3] = 3
    print(dic.items())
