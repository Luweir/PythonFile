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
import numpy as np
import pandas as pd

from Experiment.TrajectoryClusterMaster.trajCluster import approximate_trajectory_partitioning, \
    rdp_trajectory_partitioning
from Experiment.common.Point import Point
from Experiment.TrajectoryClusterMaster.trajCluster import line_segment_clustering, \
    representative_trajectory_generation
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib import pyplot as plt


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
        seg_line_x.append(seg.start.x / 100)
        seg_line_x.append(seg.end.x / 100)
        seg_line_y.append(seg.start.y / 100)
        seg_line_y.append(seg.end.y / 100)
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
def cluster_HAC(norm_cluster):
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
        for seg in cluster:
            point_list.append(seg.start)
            point_list.append(seg.end)
            data.append([seg.start.x, seg.start.y])
            data.append([seg.end.x, seg.end.y])
        data = np.array(data)
        data_zs = 1.0 * data / data.max()  # 归一化
        mergings = linkage(data_zs, method='average')

        point_index = [i for i in range(len(point_list))]
        # plt.figure(figsize=(9, 7))
        # plt.title("original data")
        # dendrogram(mergings, labels=point_index, leaf_rotation=45, leaf_font_size=8)
        # plt.show()

        # t=3 是聚成三类
        cluster_assignments = fcluster(mergings, t=3.0, criterion='maxclust')
        print(cluster_assignments)

        # 分类好
        temp_cluster_list = [[] for i in range(cluster_assignments.max())]
        for i in range(len(cluster_assignments)):
            temp_cluster_list[cluster_assignments[i] - 1].append(point_list[i])
        # 二、对于同一个类   用类中心代替类里面的点
        for ele in temp_cluster_list:
            center_x = 0
            center_y = 0
            time_dict = {}
            for e in ele:
                center_x += e.x
                center_y += e.y
                time_dict[e.trajectory_id] = e.time
            center_x = center_x / len(ele)
            center_y = center_y / len(ele)
            for e in ele:
                e.x = center_x
                e.y = center_y
            # point_cluster_result.append(cluster_point(center_x, center_y, time_dict))
        cluster_id += 1
    # return point_cluster_result


if __name__ == '__main__':
    traj1, traj2, traj3, traj4 = generateTestTrajectories()

    # part 1: partition
    part1 = approximate_trajectory_partitioning(traj1, theta=6.0, traj_id=1)
    part2 = approximate_trajectory_partitioning(traj2, theta=6.0, traj_id=2)
    part3 = approximate_trajectory_partitioning(traj3, theta=6.0, traj_id=3)
    part4 = approximate_trajectory_partitioning(traj4, theta=6.0, traj_id=4)

    # ------------------------------------  start my data testing---------------------

    epsilon = 4
    filename = "10.9.csv"
    traj1 = []  # Track points set
    data = pd.read_csv("../../data/" + filename, header=None, sep=',').values.tolist()
    start_time = data[0][0]
    for ele in data:
        traj1.append(Point(ele[1] * 100, ele[2] * 100, traj_id=1, t=int(ele[0] - start_time)))
    # print(points)
    # runPartition(traj, 6.0, 5.0)
    print("原始轨迹长度：", len(traj1))
    part1 = rdp_trajectory_partitioning(traj1, traj_id=1, epsilon=epsilon)

    filename = "10.10.csv"
    traj2 = []  # Track points set
    data = pd.read_csv("../../data/" + filename, header=None, sep=',').values.tolist()
    start_time = data[0][0]
    for ele in data:
        traj2.append(Point(ele[1] * 100, ele[2] * 100, traj_id=2, t=int(ele[0] - start_time)))
    # print(points)
    # runPartition(traj, 6.0, 5.0)
    print("原始轨迹长度：", len(traj2))
    part2 = rdp_trajectory_partitioning(traj2, 2, epsilon=epsilon)

    filename = "10.11.csv"
    traj3 = []  # Track points set
    data = pd.read_csv("../../data/" + filename, header=None, sep=',').values.tolist()
    start_time = data[0][0]
    for ele in data:
        traj3.append(Point(ele[1] * 100, ele[2] * 100, traj_id=3, t=int(ele[0] - start_time)))
    # print(points)
    # runPartition(traj, 6.0, 5.0)
    print("原始轨迹长度：", len(traj3))
    part3 = rdp_trajectory_partitioning(traj3, 3, epsilon=epsilon)

    filename = "10.12.csv"
    traj4 = []  # Track points set
    data = pd.read_csv("../../data/" + filename, header=None, sep=',').values.tolist()
    start_time = data[0][0]
    for ele in data:
        traj4.append(Point(ele[1] * 100, ele[2] * 100, traj_id=4, t=int(ele[0] - start_time)))
    # print(points)
    # runPartition(traj, 6.0, 5.0)
    print("原始轨迹长度：", len(traj4))
    part4 = rdp_trajectory_partitioning(traj4, 4, epsilon=epsilon)
    # -----------------------------------------  end my data testing -------------------
    all_segs = part1 + part2 + part3 + part4
    print("", len(all_segs))

    # -----------------------------------输出未聚类之前的 start---------------------------------------
    new_traj = []
    for ele in part1:
        new_traj.append([ele.start.time, round(ele.start.x / 100, 4), round(ele.start.y / 100), 4])
    new_traj.append([part1[-1].end.time, round(part1[-1].end.x / 100, 4), round(part1[-1].end.y / 100, 4)])

    for ele in part2:
        new_traj.append([ele.start.time, round(ele.start.x / 100, 4), round(ele.start.y / 100), 4])
    new_traj.append([part2[-1].end.time, round(part1[-1].end.x / 100, 4), round(part1[-1].end.y / 100, 4)])

    for ele in part3:
        new_traj.append([ele.start.time, round(ele.start.x / 100, 4), round(ele.start.y / 100), 4])
    new_traj.append([part3[-1].end.time, round(part1[-1].end.x / 100, 4), round(part1[-1].end.y / 100, 4)])

    for ele in part4:
        new_traj.append([ele.start.time, round(ele.start.x / 100, 4), round(ele.start.y / 100), 4])
    new_traj.append([part4[-1].end.time, round(part1[-1].end.x / 100, 4), round(part1[-1].end.y / 100, 4)])
    pd.DataFrame(new_traj).to_csv("before_clusting.csv", index=False, header=0)
    # ----------------------------------输出未聚类之前的 end--------------------------------------

    # norm_cluster, remove_cluster = line_segment_clustering(all_segs, min_lines=3, epsilon=15.0)
    norm_cluster, remove_cluster = line_segment_clustering(all_segs, min_lines=2, epsilon=10)
    for k, v in remove_cluster.items():
        print("remove cluster: the cluster %d, the segment number %d" % (k, len(v)))
    cluster_HAC(norm_cluster)

    # -------------------------------------输出文件 start -----------------------------------
    new_traj = []
    for ele in part1:
        new_traj.append([ele.start.time, round(ele.start.x / 100, 4), round(ele.start.y / 100), 4])
    new_traj.append([part1[-1].end.time, round(part1[-1].end.x / 100, 4), round(part1[-1].end.y / 100, 4)])

    for ele in part2:
        new_traj.append([ele.start.time, round(ele.start.x / 100, 4), round(ele.start.y / 100), 4])
    new_traj.append([part2[-1].end.time, round(part1[-1].end.x / 100, 4), round(part1[-1].end.y / 100, 4)])

    for ele in part3:
        new_traj.append([ele.start.time, round(ele.start.x / 100, 4), round(ele.start.y / 100), 4])
    new_traj.append([part3[-1].end.time, round(part1[-1].end.x / 100, 4), round(part1[-1].end.y / 100, 4)])

    for ele in part4:
        new_traj.append([ele.start.time, round(ele.start.x / 100, 4), round(ele.start.y / 100), 4])
    new_traj.append([part4[-1].end.time, round(part1[-1].end.x / 100, 4), round(part1[-1].end.y / 100, 4)])
    pd.DataFrame(new_traj).to_csv("output.csv", index=False, header=0)

    # -------------------------------------输出文件 end---------------------------------------

    # 作图
    cluster_s_x, cluster_s_y = [], []
    for k, v in norm_cluster.items():
        cluster_s_x.extend([s.start.x for s in v])
        cluster_s_x.extend([s.end.x for s in v])
        cluster_s_y.extend([s.start.y for s in v])
        cluster_s_y.extend([s.end.y for s in v])
        print("using cluster: the cluster %d, the segment number %d" % (k, len(v)))

    source_line_x_1 = [p.x for p in traj1]
    source_line_y_1 = [p.y for p in traj1]

    source_line_x_2 = [p.x for p in traj2]
    source_line_y_2 = [p.y for p in traj2]

    source_line_x_3 = [p.x for p in traj3]
    source_line_y_3 = [p.y for p in traj3]

    source_line_x_4 = [p.x for p in traj4]
    source_line_y_4 = [p.y for p in traj4]

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(source_line_x_1, source_line_y_1, 'g--', lw=2.0, label="trajectory 1")
    ax.scatter(source_line_x_1, source_line_y_1, c='g', alpha=0.5)
    ax.plot(source_line_x_2, source_line_y_2, 'r--', lw=2.0, label="trajectory 2")
    ax.scatter(source_line_x_2, source_line_y_2, c='r', alpha=0.5)
    ax.plot(source_line_x_3, source_line_y_3, 'b--', lw=2.0, label="trajectory 3")
    ax.scatter(source_line_x_3, source_line_y_3, c='b', alpha=0.5)
    ax.plot(source_line_x_4, source_line_y_4, 'y--', lw=2.0, label="trajectory 4")
    ax.scatter(source_line_x_4, source_line_y_4, c='y', alpha=0.5)

    for k, v in norm_cluster.items():
        for s in v:
            _x = [s.start.x, s.end.x]
            _y = [s.start.y, s.end.y]
            if s.traj_id == 1:
                ax.plot(_x, _y, c='k', lw=3.0, alpha=0.7)
            elif s.traj_id == 2:
                ax.plot(_x, _y, c='c', lw=3.0, alpha=0.7)
            elif s.traj_id == 3:
                ax.plot(_x, _y, c='m', lw=3.0, alpha=0.7)
            else:
                ax.plot(_x, _y, c='r', lw=3.0, alpha=0.7)
    ax.scatter(cluster_s_x, cluster_s_y, c='k', alpha=0.5, s=80, label="cluster")

    main_traj_dict = representative_trajectory_generation(norm_cluster, min_lines=2, min_dist=1.0)
    for c, v in main_traj_dict.items():
        v_x = [p.x for p in v]
    v_y = [p.y for p in v]
    ax.plot(v_x, v_y, 'r-', lw=4.0, label="cluster_%d_main_trajectory" % c)

    ax.legend()
    plt.savefig("./figure/trajectory-major.png", dpi=400)
    plt.show()
