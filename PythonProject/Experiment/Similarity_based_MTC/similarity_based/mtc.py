import sys

from typing import List

from Experiment.DP.dp import douglas_peucker
from Experiment.common.Point import Point
from Experiment.common.Trajectory import Trajectory


def mtc_add(t: Trajectory, c: List[Trajectory], miu):
    """
    将轨迹T 进行压缩，c表示已压缩轨迹集; T首先在c里面找最佳参考轨迹进行多轨迹压缩 然后再单轨迹压缩 哪个效果好用哪个
    :param t: 待压缩轨迹
    :param c: 已压缩轨迹集
    :param miu: 误差阈值 m
    :return: None
    """
    if len(c) == 0:
        # 如果没有参考的 直接 stc
        sample_index = dp_stc(t.points, miu / 100000)
        compressed_t = []
        for i in sample_index:
            compressed_t.append(t.points[i])
        t.points = compressed_t
        c.append(t)
    else:
        mtc_compressed_t = None
        mtc_compression_performance = sys.maxsize
        mtc_reference_traj_id = -1
        # 每个都 mtc 一遍  找到压缩率最高的
        for r in c:
            [compressed_t, len_point, len_time] = mtc(t, r, miu)
            if len_point * 2 + len_time < mtc_compression_performance:
                mtc_compressed_t = compressed_t
                mtc_compression_performance = len_point * 2 + len_time
                mtc_reference_traj_id = r.trajectory_id
        # stc 一遍  跟mtc最好的比较
        sample_index = dp_stc(t.points, miu / 100000)
        # 如果 stc 压缩效果较好 则放stc
        if mtc_compression_performance > 2 * len(sample_index):
            stc_compressed_t = []
            for i in sample_index:
                stc_compressed_t.append(t.points[i])
            t.points = stc_compressed_t
        # 否则 放 mtc
        else:
            t.points = mtc_compressed_t
            t.reference_trajectory_id = mtc_reference_traj_id
        c.append(t)


# dp 单轨迹压缩
def dp_stc(t_points, miu, just_point=True) -> list:
    t_points_copy = []
    if just_point:
        t_points_copy = t_points
    else:
        for ele in t_points:
            cur_point = ele
            while cur_point.p is not None:
                cur_point = cur_point.p
            t_points_copy.append(Point(cur_point.x, cur_point.y, ele.t))
    return douglas_peucker(t_points_copy, 0, len(t_points) - 1, miu)


def mtc(t: Trajectory, r: Trajectory, miu) -> list:
    """
    根据参考轨迹r 对轨迹t进行多轨迹压缩
    :param t: 待压缩轨迹
    :param r: 参考轨迹
    :param miu: 误差阈值
    :return: [压缩后的Point序列, 位置点的个数 , 时间映射点的个数]
    """
    # 一、t 的每个点 都在 r 中找到 匹配点，若二者在误差阈值内 则指向它的匹配点，否则 存原点
    for point in t.points:
        match_point = find_match_point(point, r.points)
        match_point.trajectory_id = r.trajectory_id
        if point.get_haversine(match_point) < miu:
            point.p = match_point
    # 二、对其中每个子段 进行单轨迹压缩 得到最后需要保留的点
    start_index = 0
    end_index = 0
    reserve_index = []  # 需要保存的点的索引
    while end_index <= len(t.points):
        if end_index < len(t.points) and ((t.points[end_index].p is None and t.points[start_index].p is None) or (
                t.points[end_index].p is not None and t.points[start_index].p is not None)):
            end_index += 1
        else:
            if end_index - start_index >= 2:
                sample_index = dp_stc(t.points[start_index:end_index],
                                      just_point=True if t.points[start_index].p is None else False, miu=miu / 100000)
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
        compressed_t.append(t.points[i])
        if t.points[i].p is not None:
            match_point_num += 1
    return [compressed_t, len(compressed_t) - match_point_num, match_point_num]


def find_match_point(t_point: Point, r_points: List[Point]) -> Point:
    """
    找r_points 中 距离 t_point 最近的点
    :param t_point: 目标点
    :param r_points:  要遍历的轨迹点集
    :return: Point 即找到的对应点
    """
    best_index = -1
    corresponding_point = []
    max_error = sys.maxsize
    index = 0
    while index < len(r_points) - 1:
        line_pre = r_points[index]
        # 如果参考轨迹的这个点 也参考了别的点 就得循环引用
        while line_pre.p is not None:
            line_pre = line_pre.p
        line_next = r_points[index + 1]
        while line_next.p is not None:
            line_next = line_next.p
        nc_point = t_point.point_intersect_line(line_pre, line_next)
        cur_dist = t_point.get_haversine(nc_point)
        if cur_dist < max_error:
            best_index = index
            corresponding_point = nc_point
        index += 1
    line_pre = r_points[best_index]
    while line_pre.p is not None:
        line_pre = line_pre.p
    line_next = r_points[best_index + 1]
    while line_next.p is not None:
        line_next = line_next.p
    delta_t = line_pre.distance(corresponding_point) / line_pre.distance(line_next) * (
            r_points[best_index + 1].t - r_points[best_index].t)
    return Point(corresponding_point.x, corresponding_point.y, t=r_points[best_index].t + delta_t)
