# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# file      :trajCluster_test.py
# target    :
# 
# output    :
# author    :Miller
# date      :2019/4/3 13:51
# log       :包含修改时间、修改人、修改line及原因
# --------------------------------------------------------------------------------
import math
import time

import numpy as np
import pandas as pd

from Experiment.TrajectoryClusterMaster.trajCluster import rdp_trajectory_partitioning, line_segment_clustering
from Experiment.common.Point import Point
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib import pyplot as plt
import Experiment.compare.compare as compare
from Experiment.common.zip import zip_compress
from Experiment.data.DataProcess.data_process import get_trajectories

EARTH_RADIUS = 6371229  # m 用于两点间距离计算


# 两点直接的距离 实际距离
def get_haversine(point_a, point_b):
    lat1 = point_a[0] * math.pi / 180
    lat2 = point_b[0] * math.pi / 180
    lon1 = point_a[1] * math.pi / 180
    lon2 = point_b[1] * math.pi / 180
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a_a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a_a), math.sqrt(1 - a_a))
    return EARTH_RADIUS * c


def euc_dist(p1, p2):
    return round(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 5)


def generateTestTrajectories():
    # ts1 = [560.0, 652.0, 543.0, 651.0, 526.0, 649.0, 510.0, 647.0, 494.0, 644.0, 477.0, 639.0, 460.0, 632.0, 446.0, 622.0, 431.0, 611.0, 417.0, 604.0, 400.0, 597.0, 383.0, 587.0, 372.0, 579.0, 363.0, 573.0, 355.0, 563.0, 356.0, 552.0, 361.0, 537.0, 370.0, 523.0, 380.0, 510.0, 391.0, 498.0, 404.0, 485.0, 415.0, 475.0, 429.0, 466.0, 444.0, 459.0, 465.0, 451.0, 493.0, 442.0, 530.0, 432.0, 568.0, 423.0, 606.0, 417.0, 644.0, 412.0, 681.0, 408.0, 714.0, 404.0, 747.0, 401.0, 770.0, 399.0, 793.0, 397.0, 818.0, 395.0]
    ts2 = [565.0, 689.0, 547.0, 682.0, 525.0, 674.0, 502.0, 668.0, 480.0, 663.0, 452.0, 660.0, 424.0, 656.0, 400.0,
           652.0,
           380.0, 650.0, 356.0, 649.0, 335.0, 647.0, 314.0, 642.0, 297.0, 639.0, 283.0, 634.0, 272.0, 625.0, 259.0,
           614.0,
           245.0, 603.0, 237.0, 596.0, 228.0, 589.0, 218.0, 582.0, 208.0, 574.0, 198.0, 567.0, 193.0, 561.0, 191.0,
           554.0,
           185.0, 551.0, 181.0, 551.0, 179.0, 549.0, 178.0, 547.0, 178.0, 544.0, 177.0, 540.0, 174.0, 533.0, 170.0,
           527.0,
           164.0, 523.0, 154.0, 521.0, 145.0, 517.0, 131.0, 514.0, 118.0, 515.0, 106.0, 515.0, 92.0, 512.0, 74.0, 507.0,
           57.0, 501.0, 40.0, 495.0, 23.0, 491.0]

    ts1 = [s - np.random.randint(10, 20) for s in ts2]

    ts3 = [s + np.random.randint(1, 10) for s in ts2]

    # [590.0, 495.0, 590.0, 498.0, 593.0, 503.0, 597.0, 507.0, 600.0, 507.0, 602.0, 505.0, 605.0, 497.0, 594.0, 487.0, 580.0, 482.0, 565.0, 483.0, 550.0, 492.0, 547.0, 497.0, 544.0, 499.0, 541.0, 494.0, 540.0, 489.0, 538.0, 479.0, 530.0, 474.0, 528.0, 485.0, 540.0, 480.0, 542.0, 477.0, 543.0, 474.0, 538.0, 476.0, 530.0, 486.0, 524.0, 497.0, 513.0, 507.0, 499.0, 516.0, 482.0, 527.0, 468.0, 538.0, 453.0, 547.0, 438.0, 555.0, 429.0, 563.0, 423.0, 566.0, 420.0, 569.0, 417.0, 572.0, 414.0, 570.0, 411.0, 566.0, 411.0, 557.0, 408.0, 545.0, 405.0, 536.0, 403.0, 530.0, 401.0, 526.0, 401.0, 522.0, 404.0, 523.0, 409.0, 523.0, 418.0, 522.0, 420.0, 522.0, 426.0, 530.0]
    ts4 = [s + np.random.randint(20, 30) for s in ts2]
    # [559.0, 492.0, 553.0, 483.0, 548.0, 475.0, 544.0, 466.0, 536.0, 456.0, 534.0, 447.0, 536.0, 438.0, 540.0, 429.0, 551.0, 419.0, 566.0, 408.0, 583.0, 399.0, 600.0, 389.0, 622.0, 379.0, 642.0, 373.0, 660.0, 373.0, 676.0, 375.0, 693.0, 381.0, 708.0, 391.0, 721.0, 402.0, 732.0, 412.0, 737.0, 421.0, 741.0, 429.0, 742.0, 437.0, 738.0, 443.0, 733.0, 447.0, 728.0, 449.0, 722.0, 450.0, 714.0, 451.0, 710.0, 445.0, 700.0, 440.0, 695.0, 440.0, 695.0, 434.0, 700.0, 435.0, 705.0, 436.0, 711.0, 435.0, 708.0, 437.0, 710.0, 440.0, 710.0, 445.0, 705.0, 455.0, 700.0, 462.0, 695.0, 470.0, 690.0, 480.0, 680.0, 490.0, 665.0, 490.0]
    traj1 = [Point(ts1[i:i + 2][0], ts1[i:i + 2][1]) for i in range(0, len(ts1), 2)]
    traj2 = [Point(ts2[i:i + 2][0], ts2[i:i + 2][1]) for i in range(0, len(ts2), 2)]
    traj3 = [Point(ts3[i:i + 2][0], ts3[i:i + 2][1]) for i in range(0, len(ts3), 2)]
    traj4 = [Point(ts4[i:i + 2][0], ts4[i:i + 2][1]) for i in range(0, len(ts4), 2)]
    return traj1, traj2, traj3, traj4


def plt_cluster_seg(norm_cluster, num):
    cluster_cur = norm_cluster[num]
    seg_line_x = []
    seg_line_y = []
    for seg in cluster_cur:
        seg_line_x.append(seg.start.x)
        seg_line_x.append(seg.end.x)
        seg_line_y.append(seg.start.y)
        seg_line_y.append(seg.end.y)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    index = 0
    while index < len(seg_line_y):
        ax.plot(seg_line_x[index:index + 2], seg_line_y[index:index + 2])
        index += 2
    plt.show()


def load_data(filename, epsolon=2):
    ...


class cluster_point:
    def __init__(self, x, y, dict):
        self.x = x
        self.y = y
        self.dict = dict


# ------------------------------------- 簇内段压缩---------------
def cluster_HAC(norm_cluster, t=3.0):
    # 最终所有聚簇 点聚类结果
    # point_cluster_result = []
    # 对于每一个聚簇：
    cluster_id = 0
    for key in norm_cluster:
        cluster = norm_cluster[key]
        if len(cluster) < 3:
            cluster_id += 1
            continue
        print("簇ID：", cluster_id, "共有 seg：", len(cluster), " 个, 点：", (2 * len(cluster)), " 个")
        # 一、运行凝聚型层次聚类  将簇内所有点凝聚聚类   保证类内误差距离
        point_list = []
        data = []
        # 注意可能有连续段  到时候还要用集合!!!!!l
        for seg in cluster:
            point_list.append(seg.start)
            point_list.append(seg.end)
            data.append([seg.start.x, seg.start.y])
            data.append([seg.end.x, seg.end.y])
        data = np.array(data)
        # print(data)
        # data_zs = 1.0 * data / data.max()  # 归一化
        mergings = linkage(data, method='complete', metric="euclidean")

        point_index = [i for i in range(len(point_list))]
        # plt.figure(figsize=(9, 7))
        # plt.title("original data")
        # dendrogram(mergings, labels=point_index, leaf_rotation=45, leaf_font_size=8)
        # plt.show()

        # t 是标准   如t=3 是聚成三类  或者是后面距离的标准
        cluster_assignments = fcluster(mergings, t=t, criterion='distance')
        print("该聚簇的点可分为：", cluster_assignments.max(), " 类")
        # print("cluster_assignments:", cluster_assignments)

        # 分类
        temp_cluster_list = [[] for i in range(cluster_assignments.max())]
        for i in range(len(cluster_assignments)):
            temp_cluster_list[cluster_assignments[i] - 1].append(point_list[i])
        # 二、对于同一个类   用类中心代替类里面的点
        for ele in temp_cluster_list:
            center_x = 0
            center_y = 0
            t_dict = {}
            for e in ele:
                center_x += e.x
                center_y += e.y
                t_dict[e.trajectory_id] = e.t
            center_x = center_x / len(ele)
            center_y = center_y / len(ele)
            for e in ele:
                e.x = center_x
                e.y = center_y
            # point_cluster_result.append(cluster_point(center_x, center_y, t_dict))
        cluster_id += 1
    # return point_cluster_result


def generate_trajectory(data):
    trajectory = []
    for i in range(len(data)):
        # trajectory.append(Point(data.iloc[i][1] + random.uniform(-0.01, 0.01),
        #                         data.iloc[i][2] + random.uniform(-0.01, 0.01),
        #                         data.iloc[i][0] + random.randint(-5, 5)))
        trajectory.append(Point(data.iloc[i][1],
                                data.iloc[i][2],
                                data.iloc[i][0]))
    return trajectory


def run():
    res = []
    # epsilon = 110
    for i in range(1, 50):
        epsilon = i * 20
        trajectories = []
        parts = []
        path = r'E:\Desktop\Programmer\PythonFile\PythonProject\Experiment\data\SyntheticData'
        for i in range(7):
            trajectory = []  # Track points set
            data = pd.read_csv(path + r"\output_origin_trajectory_" + str(i) + ".csv", header=None,
                               sep=',').values.tolist()
            for ele in data:
                trajectory.append(Point(x=ele[1], y=ele[2], trajectory_id=i, t=int(ele[0])))
            print("原始轨迹长度：", len(trajectory))
            part = rdp_trajectory_partitioning(trajectory, traj_id=i, epsilon=epsilon / 100000)
            trajectories.append(trajectory)
            parts.append(part)
        # -----------------------------------------  end my data testing -------------------
        all_segs = []
        for ele in parts:
            all_segs += ele
        print("一共多少轨迹段:", len(all_segs))
        # 进行点的聚类
        compress_start_time = time.perf_counter()

        # norm_cluster, remove_cluster = line_segment_clustering(all_segs, min_lines=3, epsilon=15.0)
        norm_cluster, remove_cluster = line_segment_clustering(all_segs, min_lines=2, epsilon=0.5)
        for k, v in remove_cluster.items():
            print("remove cluster: the cluster %d, the segment number %d" % (k, len(v)))
        cluster_HAC(norm_cluster, t=epsilon / 100000.0)
        compress_end_time = time.perf_counter()
        # -------------------------------------输出聚类之后的 start -----------------------------------
        new_traj = []
        for part in parts:
            for ele in part:
                new_traj.append([int(ele.start.t), round(ele.start.x, 4), round(ele.start.y, 4)])
            new_traj.append([int(part[-1].end.t), round(part[-1].end.x, 4), round(part[-1].end.y, 4)])
        pd.DataFrame(new_traj).to_csv("mtc_2cluster_compressed_trajectory.txt", index=False, header=0)
        # -------------------------------------输出聚类之后的 end---------------------------------------
        old_trajectories = get_trajectories(trajectory_type="point_list")
        average_ped_error = 0
        max_ped_error = 0
        average_sed_error = 0
        max_sed_error = 0
        average_speed_error = 0
        max_speed_error = 0
        average_angle_error = 0
        max_angle_error = 0

        for i in range(len(old_trajectories)):
            new_traj = []
            for ele in parts[i]:
                new_traj.append(Point(x=round(ele.start.x, 4), y=round(ele.start.y, 4), t=int(ele.start.t)))
            new_traj.append(
                Point(x=round(parts[i][-1].end.x, 4), y=round(parts[i][-1].end.y, 4), t=int(parts[i][-1].end.t)))
            [a, b] = compare.get_PED_error(old_trajectories[i], new_traj)
            [c, d] = compare.get_SED_error(old_trajectories[i], new_traj)
            [e, f] = compare.get_speed_error(old_trajectories[i], new_traj)
            [g, h] = compare.get_angle_error(old_trajectories[i], new_traj)
            average_ped_error += a
            max_ped_error = max(max_ped_error, b)
            average_sed_error += c
            max_sed_error = max(max_sed_error, d)
            average_speed_error += e
            max_speed_error = max(max_speed_error, f)
            average_angle_error += g
            max_angle_error = max(max_angle_error, h)

        print("average_ped_error:", average_ped_error / len(old_trajectories))
        print("max_ped_error:", max_ped_error)
        print("average_sed_error:", average_sed_error / len(old_trajectories))
        print("max_sed_error:", max_sed_error)
        print("average_speed_error:", average_speed_error / len(old_trajectories))
        print("max_speed_error:", max_speed_error)
        print("average_angle_error:", average_angle_error / len(old_trajectories))
        print("max_speed_error:", max_angle_error)
        [a, b] = zip_compress("mtc_2cluster_compressed_trajectory.txt")
        res.append(
            [epsilon, average_ped_error / len(trajectories), max_ped_error, average_sed_error / len(trajectories),
             max_sed_error, average_speed_error / len(trajectories), max_speed_error,
             average_angle_error / len(trajectories), max_angle_error, a, b, (compress_end_time - compress_start_time)])
    res = pd.DataFrame(res,
                       columns=['误差阈值', '平均ped误差', '最大ped误差', '平均sed误差', '最大sed误差', '平均速度误差', '最大速度误差', '平均角度误差',
                                '最大角度误差', '压缩后文件大小', 'zip后文件大小', '压缩时间(s)'])
    return res


if __name__ == '__main__':
    # traj1, traj2, traj3, traj4 = generateTestTrajectories()

    # part 1: partition
    # part1 = approximate_trajectory_partitioning(traj1, theta=6.0, traj_id=1)
    # part2 = approximate_trajectory_partitioning(traj2, theta=6.0, traj_id=2)
    # part3 = approximate_trajectory_partitioning(traj3, theta=6.0, traj_id=3)
    # part4 = approximate_trajectory_partitioning(traj4, theta=6.0, traj_id=4)

    # ------------------------------------  start my data testing---------------------
    res = run()
