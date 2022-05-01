import time

import numpy as np
import math
import Experiment.DP.dp as dp
import Experiment.compare_result.compare as cr


class FeaturePoints:
    lat = 0
    lon = 0
    idToTime = {}

    def __init__(self, lat=0, lon=0):
        self.lat = lat
        self.lon = lon


if __name__ == '__main__':
    filename = "10.9.csv"
    epsilon = 0.01  # 0.0001
    save_filename = "result.csv"
    points = dp.gps_reader(filename)
    start_time = time.perf_counter()
    sample_index = dp.td_tr(points, 0, len(points) - 1, epsilon)
    sample = []
    for e in sample_index:
        sample.append(points[e])
    end_time = time.perf_counter()
    print("DP-TR")
    print("dataset:" + str(filename))
    cr.get_CR_and_time(save_filename, start_time, end_time, points, sample)
    print("PED距离误差：" + str(cr.get_PED_error(points, sample)))
    print("SED距离误差：" + str(cr.get_SED_error(points, sample)))
    print("Angle角度误差：" + str(cr.get_angle_error2(points, sample)))
    cr.get_dtw(points, sample)
