import pandas as pd
import numpy as np
import time
import math
import Experiment.compare_result.compare as cr
import Experiment.src.trajectory_graph as tg
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


# 加载数据集
def gps_reader(filename):
    points = []  # Track points set
    data = pd.read_csv("../data/" + filename, header=None, sep=',').values.tolist()
    for ele in data:
        points.append([ele[0], ele[1], ele[2]])
    return points


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


# 计算两点之间的距离
def get_distance(point_a, point_b):
    return get_haversine(point_a, point_b)  # 距离单位：米


# 获得两点之间的速度
def get_speed(point_a, point_b):
    return get_distance(point_a, point_b) / (point_b[0] - point_a[0])


# 获得两点之间的角度
def get_angel(point_a, point_b):
    lat_diff = point_b[1] - point_a[1]
    lon_diff = point_b[2] - point_a[2]
    res = math.atan2(lon_diff, lat_diff)
    # res = res * 180 / math.pi
    return res  # 得到的是弧度制， x*Math.PI/180 => 角度


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


def run(filename):
    epsilon = 0.01  # 0.0001
    save_filename = "result.csv"
    points = gps_reader(filename)
    start_time = time.perf_counter()
    sample_index = td_tr(points, 0, len(points) - 1, epsilon)
    sample = []
    for e in sample_index:
        sample.append(points[e])
    end_time = time.perf_counter()
    print("DP-TR")
    print("dataset:" + str(filename))
    cr.get_CR_and_time(save_filename, start_time, end_time, points, sample)
    print("PED距离误差：" + str(cr.get_PED_error(points, sample)))
    print("SED距离误差：" + str(cr.get_SED_error(points, sample)))
    print("平均Angle角度误差：" + str(cr.get_angle_error2(points, sample)))
    print("平均Speed速度误差：" + str(cr.get_speed_error(points, sample)))
    # cr.get_dtw(points, sample)
    # tg.get_two_line_chart(pd.DataFrame(points, columns=['time', 'longitude', 'latitude']),
    #                       pd.DataFrame(sample, columns=['time', 'longitude', 'latitude']))

    feature = []
    for i in range(len(sample)):
        if i == len(sample) - 1:
            # time      longitude      latitude        velocity       angle
            feature.append([sample[i][0], 0, 0])
        else:
            feature.append(
                [sample[i][0], get_speed(sample[i], sample[i + 1]),
                 get_angel(sample[i], sample[i + 1])])

    return points, sample, feature


def get_speed_or_angle_dtw(flag, f1, f2, f1l, f1r, f2l, f2r):
    print(flag + " F1:" + str(f1l) + "-" + str(f1r) + " F2:" + str(f2l) + "-" + str(f2r) + " dtw：" + str(
        cr.get_dtw(f1, f2)))


def get_speed_or_angle_pearson(flag, f1, f2, f1l, f1r, f2l, f2r):
    r, p = stats.pearsonr(f1, f2)
    print(flag + " F1:" + str(f1l) + "-" + str(f1r) + " F2:" + str(f2l) + "-" + str(f2r) + " r：" + str(r))


def fast_segmentation1(feature1, feature2, f1l, f1r, f2l, f2r):
    mid1 = int((f1l + f1r) / 2)
    mid2 = int((f2l + f2r) / 2)
    # get_speed_or_angle_pearson('V', feature1[f1l:f1l + min(mid1 - f1l, mid2 - f2l), 1],
    #                            feature2[f2l:f2l + min(mid1 - f1l, mid2 - f2l), 1], f1l, mid1, f2l, mid2)
    # get_speed_or_angle_pearson('A', feature1[f1l:f1l + min(mid1 - f1l, mid2 - f2l), 2],
    #                            feature2[f2l:f2l + min(mid1 - f1l, mid2 - f2l), 2], f1l, mid1, f2l, mid2)
    # get_speed_or_angle_pearson('V', feature1[mid1:mid1 + min(f1r - mid1, f2r - mid2), 1],
    #                            feature2[mid2:mid2 + min(f1r - mid1, f2r - mid2), 1], mid1, f1r, mid2, f2r)
    # get_speed_or_angle_pearson('A', feature1[mid1:mid1 + min(f1r - mid1, f2r - mid2), 2],
    #                            feature2[mid2:mid2 + min(f1r - mid1, f2r - mid2), 2], mid1, f1r, mid2, f2r)
    get_speed_or_angle_dtw('V', feature1[f1l:mid1, 0:2],
                           feature2[f2l:mid2, 0:2], f1l, mid1, f2l, mid2)
    get_speed_or_angle_dtw('A', feature1[f1l:mid1, [0, 2]],
                           feature2[f2l:mid2, [0, 2]], f1l, mid1, f2l, mid2)
    get_speed_or_angle_dtw('V', feature1[mid1 + 1:f1r, 0:2],
                           feature2[mid2 + 1:f2r, 0:2], mid1 + 1, f1r, mid2 + 1, f2r)
    get_speed_or_angle_dtw('A', feature1[mid1 + 1:f1r, [0, 2]],
                           feature2[mid2 + 1:f2r, [0, 2]], mid1 + 1, f1r, mid2 + 1, f2r)
    # 点数过小  交叉检测dtw
    if f1r - f1l < 10 or f2r - f2l < 10:
        get_speed_or_angle_dtw('V', feature1[f1l:mid1, 0:2],
                               feature2[mid2 + 1:f2r, 0:2], f1l, mid1, mid2 + 1, f2r)
        get_speed_or_angle_dtw('A', feature1[f1l:mid1, [0, 2]],
                               feature2[mid2 + 1:f2r, [0, 2]], f1l, mid1, mid2 + 1, f2r)
        get_speed_or_angle_dtw('V', feature1[mid1 + 1:f1r, 0:2],
                               feature2[f2l:mid2, 0:2], mid1 + 1, f1r, f2l, mid2)
        get_speed_or_angle_dtw('A', feature1[mid1 + 1:f1r, [0, 2]],
                               feature2[f2l:mid2, [0, 2]], mid1 + 1, f1r, f2l, mid2)

        # 交叉判断相似度
        return

    fast_segmentation1(feature1, feature2, f1l, mid1, f2l, mid2)
    fast_segmentation1(feature1, feature2, mid1 + 1, f1r, mid2 + 1, f2r)


if __name__ == '__main__':
    points1, sample1, feature1 = run("10.9.csv")
    points2, sample2, feature2 = run("10.10.csv")

    # tg.get_two_line_chart(pd.DataFrame(sample1, columns=['time', 'longitude', 'latitude']),
    #                       pd.DataFrame(sample2, columns=['time', 'longitude', 'latitude']))

    f1l = 0
    f1r = len(feature1) - 1
    f2l = 0
    f2r = len(feature2) - 1
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    # print(feature1)
    # print(feature2)
    # fast_segmentation1(feature1, feature2, 0, len(feature1) - 1, 0, len(feature2) - 1)

    # 计算相关性
    sample1_left = 0
    sample1_right = 10
    velocity_relate = []
    while sample1_right < len(feature1):
        sample2_left = 0
        sample2_right = 10
        while sample2_right < len(feature2):
            # x = np.array(feature1[sample1_left:sample1_right, 1]).reshape(-1, 1)
            # y = np.array(feature2[sample2_left:sample2_right, 1]).reshape(-1, 1)
            # model = LinearRegression()
            # model.fit(x, y)
            # y_prd = model.predict(x)
            # n = 10
            # Regression = sum((y_prd - np.mean(y)) ** 2)  # 回归平方和
            # Residual = sum((y - y_prd) ** 2)  # 残差平方和
            # total = sum((y - np.mean(y)) ** 2)  # 总体平方和
            # R_square = 1 - Residual / total  # 决定系数R^2
            # print(R_square)
            # message0 = '一元线性回归方程为: ' + '\ty' + '=' + str(model.intercept_) + ' + ' + str(model.coef_[0]) + '*x'
            # print(message0)
            # print(sample1_left, sample1_right, sample2_left, sample2_right, r, p)
            r, p = stats.pearsonr(feature1[sample1_left:sample1_right, 1], feature2[sample2_left:sample2_right, 1])
            if p < 0.001:
                velocity_relate.append([sample1_left, sample1_right, sample2_left, sample2_right])
            # if R_square > 0.8:
            #     velocity_relate.append([sample1_left, sample1_right, sample2_left, sample2_right])
            sample2_right += 1
            sample2_left += 1
        sample1_right += 1
        sample1_left += 1

    sample1_left = 0
    sample1_right = 10
    angle_relate = []
    while sample1_right < len(feature1):
        sample2_left = 0
        sample2_right = 10
        while sample2_right < len(feature2):
            # x = np.array(feature1[sample1_left:sample1_right, 2]).reshape(-1, 1)
            # y = np.array(feature2[sample2_left:sample2_right, 2]).reshape(-1, 1)
            # model = LinearRegression()
            # model.fit(x, y)
            # y_prd = model.predict(x)
            # n = 10
            # Regression = sum((y_prd - np.mean(y)) ** 2)  # 回归平方和
            # Residual = sum((y - y_prd) ** 2)  # 残差平方和
            # total = sum((y - np.mean(y)) ** 2)  # 总体平方和
            # R_square = 1 - Residual / total  # 决定系数R^2
            # print(R_square)
            # message0 = '一元线性回归方程为: ' + '\ty' + '=' + str(model.intercept_) + ' + ' + str(model.coef_[0]) + '*x'
            # print(message0)
            r, p = stats.pearsonr(feature1[sample1_left:sample1_right, 2], feature2[sample2_left:sample2_right, 2])
            print(sample1_left, sample1_right, sample2_left, sample2_right, r, p)
            if p < 0.001:
                angle_relate.append([sample1_left, sample1_right, sample2_left, sample2_right])
            # if R_square > 0.8:
            #     angle_relate.append([sample1_left, sample1_right, sample2_left, sample2_right])
            sample2_right += 1
            sample2_left += 1
        sample1_right += 1
        sample1_left += 1

    print("velocity_relate")
    print(velocity_relate)
    print("angle_relate")
    print(angle_relate)
    print("intersection")
    intersection = [i for i in velocity_relate if i in angle_relate]
    print(intersection)
    i = intersection[0][2]
    j = intersection[0][0]
    model1 = LinearRegression()
    model2 = LinearRegression()
    model1.fit(np.array(feature1[intersection[0][0]:intersection[0][1], 1]).reshape(-1, 1),
               np.array(feature2[intersection[0][2]:intersection[0][3], 1]).reshape(-1, 1))
    model2.fit(np.array(feature1[intersection[0][0]:intersection[0][1], 2]).reshape(-1, 1),
               np.array(feature2[intersection[0][2]:intersection[0][3], 2]).reshape(-1, 1))

    origin_value = feature2[intersection[0][2]:intersection[0][3], :]
    y1_prd = model1.predict(np.array(feature1[intersection[0][0]:intersection[0][1], 1]).reshape(-1, 1))
    y2_prd = model2.predict(np.array(feature1[intersection[0][0]:intersection[0][1], 2]).reshape(-1, 1))
    cur_value = []
    for i in range(len(y1_prd)):
        cur_value.append([origin_value[i][0], y1_prd[i][0], y2_prd[i][0]])

    origin_value = pd.DataFrame(origin_value)
    cur_value = pd.DataFrame(cur_value)
