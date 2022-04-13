#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from numpy import array


# 熵权法1：仅针对于全部都是正效应的指标
# 定义熵值法函数
# def cal_weight(x):
#     """熵值法计算变量的权重"""
#     # 标准化
#     x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))
#
#     # 求k
#     rows = x.index.size  # 行
#     cols = x.columns.size  # 列
#     k = 1.0 / math.log(rows)
#
#     lnf = [[None] * cols for i in range(rows)]
#
#     # 矩阵计算--
#     # 信息熵
#     # p=array(p)
#     x = array(x)
#     lnf = [[None] * cols for i in range(rows)]
#     lnf = array(lnf)
#     for i in range(0, rows):
#         for j in range(0, cols):
#             if x[i][j] == 0:
#                 lnfij = 0.0
#             else:
#                 p = x[i][j] / x.sum(axis=0)[j]
#                 lnfij = math.log(p) * p * (-k)
#             lnf[i][j] = lnfij
#     lnf = pd.DataFrame(lnf)
#     E = lnf
#
#     # 计算冗余度
#     d = 1 - E.sum(axis=0)
#     # 计算各指标的权重
#     w = [[None] * 1 for i in range(cols)]
#     for j in range(0, cols):
#         wj = d[j] / sum(d)
#         w[j] = wj
#         # 计算各样本的综合得分,用最原始的数据
#
#     w = pd.DataFrame(w)
#     return w, x

# 熵权法2：既可以处理正效应变量也可以处理负效应变量
# 地址：https://blog.csdn.net/ziyin_2013/article/details/116496411

def entropy_weight_method2(filename):
    data = pd.read_excel(filename)
    indicator = data.columns.tolist()  # 指标个数
    project = data.index.tolist()  # 方案数、评价主体
    value = data.values
    # 数据标准化
    flag = ["+", "+", "-", "-", "-", "-", "+", "+", "+", "+", "+"]  # 表示指标为正向指标还是反向指标
    std_value = std_data(value, flag, indicator)
    std_value.round(3)
    # 结果
    w = cal_weight(indicator, project, std_value)
    w = pd.DataFrame(w, index=data.columns, columns=['权重'])
    print("#######权重:#######")
    print(w)
    w.to_excel("WeightTable.xlsx")
    score = np.dot(std_value, w).round(5)
    score = score.tolist()
    res = []
    temp = []
    for i in range(1, len(score) + 1):
        temp.append(score[i - 1])
        if i % 7 == 0:
            res.append(temp)
            temp = []
    pd.DataFrame(res).to_excel("score.xlsx")
    return res


# 标准化数据
def std_data(value, flag, indicator):
    for i in range(len(indicator)):
        if flag[i] == '+':
            value[:, i] = (value[:, i] - np.min(value[:, i], axis=0)) / (
                    np.max(value[:, i], axis=0) - np.min(value[:, i], axis=0)) + 0.01
        elif flag[i] == '-':
            value[:, i] = (np.max(value[:, i], axis=0) - value[:, i]) / (
                    np.max(value[:, i], axis=0) - np.min(value[:, i], axis=0)) + 0.01
    return value


# 定义熵值法函数、熵值法计算变量的权重
def cal_weight(indicator, project, value):
    p = np.array([[0.0 for i in range(len(indicator))] for i in range(len(project))])
    # print(p)
    for i in range(len(indicator)):
        p[:, i] = value[:, i] / np.sum(value[:, i], axis=0)

    e = -1 / np.log(len(project)) * sum(p * np.log(p))  # 计算熵值
    g = 1 - e  # 计算一致性程度
    w = g / sum(g)  # 计算权重
    return w


if __name__ == '__main__':
    result = entropy_weight_method2("dataY.xlsx")
