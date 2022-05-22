import time
import math
import Experiment.compare_result.compare as cr
import pandas as pd


# 加载数据集
def gps_reader(filename):
    points = []  # Track points set
    data = pd.read_csv("../data/" + filename, header=None, sep=',').values.tolist()
    for ele in data:
        points.append([ele[0], ele[1], ele[2]])
    return points


def get_ped(s, m, e):
    a = e[2] - s[2]
    b = s[1] - e[1]
    c = e[1] * s[2] - s[1] * e[2]
    if a == 0 and b == 0:
        return 0
    short_dist = abs((a * m[1] + b * m[2] + c) / math.sqrt(a * a + b * b))
    return short_dist


def get_sed(s, m, e):
    numerator = m[0] - s[0]
    denominator = e[0] - s[0]
    time_ratio = 1
    if denominator != 0:
        time_ratio = numerator / denominator
    lat = s[1] + (e[1] - s[1]) * time_ratio
    lon = s[2] + (e[2] - s[2]) * time_ratio
    lat_diff = lat - m[1]
    lon_diff = lon - m[2]
    return math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)


# dp
def douglas_peucker(points, start, last, epsilon):
    d_max = 0
    index = start
    rec_result = []
    for i in range(start + 1, last):
        d = get_ped(points[start], points[i], points[last])
        if d > d_max:
            index = i
            d_max = d
    if d_max > epsilon:
        rec_result1 = douglas_peucker(points, start, index, epsilon)
        rec_result2 = douglas_peucker(points, index, last, epsilon)
        rec_result.extend(rec_result1)
        rec_result.extend(rec_result2[1:])
    else:
        rec_result.append(start)
        rec_result.append(last)
    return rec_result


# TD-TR
def td_tr(points, start, last, epsilon):
    d_max = 0
    index = start
    rec_result = []
    for i in range(start + 1, last):
        d = get_sed(points[start], points[i], points[last])
        if d > d_max:
            index = i
            d_max = d
    if d_max > epsilon:
        rec_result1 = td_tr(points, start, index, epsilon)
        rec_result2 = td_tr(points, index, last, epsilon)
        rec_result.extend(rec_result1)
        # the first point must can't join because it equals with the last point of rec_result1
        rec_result.extend(rec_result2[1:])
    else:
        rec_result.append(start)
        rec_result.append(last)
    return rec_result


if __name__ == '__main__':
    filename = "10.9.csv"
    epsilon = 0.01  # 0.0001
    save_filename = "result.csv"
    points = gps_reader(filename)
    start_time = time.perf_counter()
    sample_index = douglas_peucker(points, 0, len(points) - 1, epsilon)
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
    # tg.get_two_line_chart(pd.DataFrame(points, columns=['time', 'longitude', 'latitude']),
    #                       pd.DataFrame(sample, columns=['time', 'longitude', 'latitude']))
