from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

from Experiment.data.data_process import get_trajectories
if __name__ == '__main__':
    t = 20
    trajectories = get_trajectories(trajectory_type="point_list")
    data = []
    for i in range(len(trajectories)):
        for ele in trajectories[i]:
            data.append([ele.x, ele.y])

    data = np.array(data)

    # 以下代码为仅使用层次聚类
    mergings = linkage(data, method='complete', metric="euclidean")
    # plt.figure(figsize=(9, 7))
    # plt.title("original data")
    # point_index = [i for i in range(len(data))]
    # dendrogram(mergings, labels=point_index, leaf_rotation=45, leaf_font_size=3)
    # plt.show()

    cluster_assignments = fcluster(mergings, t=t / 100000.0, criterion='distance')
    print(cluster_assignments)
    for i in range(cluster_assignments.max()):
        print('cluster', i, ':')
        num = 1
        for index, value in enumerate(cluster_assignments):
            if value == i:
                if num % 5 == 0:
                    print()
                num += 1
                print(data[index], end='  ')
        print()
