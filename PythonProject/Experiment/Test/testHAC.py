from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Point:
    def __init__(self, pointId, lat, lon):
        self.pointId = pointId
        self.lat = lat
        self.lon = lon


data = pd.read_csv('../data/AirlineData/10.9.csv', header=None)
data = data[:20]

pointList = [i + 1 for i in range(20)]
data[0] = pointList

data.pop(0)
data = np.array(data)
data_zs = 1.0 * data / data.max()  # 归一化

# 以下代码为仅使用层次聚类
plt.figure(figsize=(9, 7))
plt.title("original data")
mergings = linkage(data_zs, method='average')
dendrogram(mergings, labels=pointList, leaf_rotation=45, leaf_font_size=8)
plt.show()

cluster_assignments = fcluster(mergings, t=3.0, criterion='maxclust')
print(cluster_assignments)
for i in range(1, 4):
    print('cluster', i, ':')
    num = 1
    for index, value in enumerate(cluster_assignments):
        if value == i:
            if num % 5 == 0:
                print()
            num += 1
            print(pointList[index], end='  ')
    print()

#
# # 以下代码为加入PCA进行对比
# class myPCA():
#
#     def __init__(self, X, d=2):
#         self.X = X
#         self.d = d
#
#     def mean_center(self, data):
#         """
#         去中心化
#         :param data: data sets
#         :return:
#         """
#         n, m = data.shape
#         for i in range(m):
#             aver = np.sum(self.X[:, i]) / n
#             x = np.tile(aver, (1, n))
#             self.X[:, i] = self.X[:, i] - x
#
#     def runPCA(self):
#         # 计算协方差矩阵，得到特征值，特征向量
#         S = np.dot(self.X.T, self.X)
#         S_val, S_victors = np.linalg.eig(S)
#         index = np.argsort(-S_val)[0:self.d]
#         Y = S_victors[:, index]
#         # 得到输出样本集
#         Y = np.dot(self.X, Y)
#         return Y
#
#
# data_for_pca = np.array(data_zs)
# pcaObject = myPCA(data_for_pca, d=2)
# pcaObject.mean_center(data_for_pca)
# res = pcaObject.runPCA()
# --------------------------------------------------
# plt.figure(figsize=(9, 7))
# plt.title("after pca")
# mergings = linkage(res,method='average')
# print(mergings)
# dendrogram(mergings,labels=country,leaf_rotation=45,leaf_font_size=8)
# plt.show()

# cluster_assignments = fcluster(mergings, t=3.0, criterion='maxclust')
# print(cluster_assignments)
# for i in range(1, 4):
#     print('cluster', i, ':')
#     num = 1
#     for index, value in enumerate(cluster_assignments):
#         if value == i:
#             if num % 5 == 0:
#                 print()
#             num += 1
#             print(country[index], end='  ')
#     print()
