import sys

import Experiment.Similarity_based_MTC.similarity_based.Trajectory as Trajectory
from Experiment.DP.dp import douglas_peucker
from Experiment.Similarity_based_MTC.similarity_based.Point import Point
from Experiment.TrajStore.src.cluster import euc_dist, point_intersect_line, get_haversine


# dp 单轨迹压缩
def dp_stc(t_points, miu, use_point=True):
    point_list = []
    if use_point:
        for ele in t_points:
            point_list.append([ele.t, ele.x, ele.y])
    else:
        for ele in t_points:
            point_list.append([ele.t, ele.t2.x, ele.t2.y])
    return douglas_peucker(point_list, 0, len(point_list) - 1, miu)


def mtc(t: Trajectory, r: Trajectory, miu):
    r_points = r.to_list()
    # 一、t 的每个点 都在 r 中找到 匹配点，若二者在误差阈值内 则指向它的匹配点，否则 存原点
    for point in t.points:
        match_point = find_match_point1(point.to_list(), r_points)
        if get_haversine(match_point.to_list(), point.to_list()) < miu:
            point.t2 = match_point
    # 二、对其中每个子段 进行单轨迹压缩 得到最后需要保留的点
    t_points = t.points  # t_points=[ point1, point2 ......]
    start_index = 0
    end_index = 0
    reserve_index = []  # 需要保存的点的索引
    while end_index <= len(t_points):
        if end_index < len(t_points) and t_points[end_index].t2 == t_points[start_index].t2:
            end_index += 1
        else:
            if end_index - start_index >= 2:
                sample_index = dp_stc(t_points[start_index:end_index],
                                      use_point=True if t_points[start_index].t2 is None else False, miu=miu)
                for ele in sample_index:
                    reserve_index.append(ele + start_index)
            else:
                for i in range(start_index, end_index):
                    reserve_index.append(i)
            start_index = end_index
            end_index = start_index + 1
    # 压缩后的点集
    compressed_t = []
    # 转换为参考轨迹的时间 的点的数量 （相同压缩轨迹长度下 它越多 后期压缩率越高  或者 根据自身点数量*2+参考点数量*1进行比较）
    match_point_num = 0
    for i in reserve_index:
        compressed_t.append(t_points[i])
        if t_points[i].t2 is not None:
            match_point_num += 1
    return [compressed_t, len(compressed_t) - match_point_num, match_point_num]


# 直接认为最近点就是最佳匹配点
# t_point[x,y]  t_points[[t1,x1,y1],[t2,x2,y2]...]
def find_match_point1(t_point: list, r_points: list):
    best_index = -1
    corresponding_point = []
    max_error = sys.maxsize
    index = 0
    while index < len(r_points) - 1:
        line_pre = r_points[index]
        line_next = r_points[index + 1]
        nc_point = point_intersect_line(t_point, line_pre[1:], line_next[1:])
        cur_dist = get_haversine(t_point, nc_point)
        if cur_dist < max_error:
            best_index = index
            corresponding_point = nc_point
    line_pre = r_points[best_index]
    line_next = r_points[best_index + 1]
    delta_t = euc_dist(line_pre, corresponding_point) / euc_dist(line_pre, line_next) \
              * (line_next[0] - line_pre[0])
    return Point(corresponding_point[0], corresponding_point[1], line_pre[2] + delta_t)


# 根据最近点 和 半径长度找最佳映射点
def find_match_point2(t_point: list, t_radis, r_points: list):
    res_point = Point(-1, -1, -1)
    # 隔 t_point 最近的两个点 需要存储  防止 r 中找不到对应点
    nc_point_closed_index = 0
    nc_point_closed_error = sys.maxsize
    # 正式开始
    index = 0
    index_set = set()
    min_haversine_dist = sys.maxsize
    while index < len(r_points) - 1:
        pre_dist = euc_dist(r_points[index], r_points[0])
        aft_dist = euc_dist(r_points[index + 1], r_points[0])
        # 更新
        if abs((pre_dist + aft_dist) / 2 - t_radis) < nc_point_closed_error:
            nc_point_closed_index = index
            nc_point_closed_error = abs((pre_dist + aft_dist) / 2 - t_radis)
        # 线性距离在 两点之间  为了保证精度 每个对应点位置的确定都遍历一遍 t2
        if (pre_dist <= t_radis <= aft_dist) or (pre_dist >= t_radis >= aft_dist):
            index_set.add(index)
        index += 1
    # 将最后一个点 连接 起点 找可能存在的一个映射点 => 即伪造这么一个线段（假设它最后回到了起点） 加入lndex_set中 不会出现找不到映射点问题 解决 映射点精度过低问题
    pre_dist = euc_dist(r_points[index], r_points[0])
    aft_dist = 0
    if pre_dist >= t_radis >= aft_dist:
        index_set.add(index)
    if index == len(r_points) - 1 and len(index_set) == 0:
        line_pre = r_points[nc_point_closed_index]
        line_next = r_points[nc_point_closed_index + 1]
        nc_point = point_intersect_line(t_point, line_pre, line_next)
        dist1 = get_haversine(nc_point, t_point)
        dist2 = get_haversine(t_point, r_points[index])
        # 如果非最佳匹配点 隔 t1_point 距离 小于 最后一个点 隔  t1_point 的距离 那就存前者
        if dist1 < dist2:
            min_haversine_dist = dist1
            delta_t = euc_dist(line_pre.to_list(), nc_point) / euc_dist(line_pre.to_list(),
                                                                        line_next.to_list()) * (
                              line_next.t - line_pre.t)
            res_point.t = line_pre[2] + int(delta_t)
        # 否则就存后者
        else:
            min_haversine_dist = dist2
            res_point.t = r_points[index][2]
    else:
        ...
