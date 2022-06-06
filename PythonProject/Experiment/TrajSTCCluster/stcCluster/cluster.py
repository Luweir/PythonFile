import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster


def cluster_HAC(trajectories, t=3.0):
    # 最终所有聚簇 点聚类结果
    # point_cluster_result = []
    # 对于每一个聚簇：

    # 一、运行凝聚型层次聚类  将所有点凝聚聚类   保证类内误差距离
    point_list = []
    data = []
    # 注意可能有连续段  到时候还要用集合!!!!!l
    for trajectory in trajectories:
        for point in trajectory:
            point_list.append(point)
            data.append([point.x, point.y])
    data = np.array(data)
    print("共有特征点：", len(point_list), "个")

    mergings = linkage(data, method='complete', metric="euclidean")

    # point_index = [i for i in range(len(point_list))]
    # plt.figure(figsize=(9, 7))
    # plt.title("original data")
    # dendrogram(mergings, labels=point_index, leaf_rotation=45, leaf_font_size=8)
    # plt.show()

    # t 是标准   如t=3 是聚成三类  或者是后面距离的标准
    cluster_assignments = fcluster(mergings, t=t, criterion='distance')
    print("所有特征点可分为：", cluster_assignments.max(), " 类")
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
        if len(ele) == 1:
            continue
        for e in ele:
            center_x += e.x
            center_y += e.y
            t_dict[e.trajectory_id] = e.t
        center_x = center_x / len(ele)
        center_y = center_y / len(ele)
        for e in ele:
            e.x = center_x
            e.y = center_y
