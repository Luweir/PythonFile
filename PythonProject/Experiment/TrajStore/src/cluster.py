import numpy as np
import pandas as pd
import math
import sys


# ---------------------------------------------
# author: luweir
# target: TrajStore's trajectory cluster
# date: 2022-5-13
# ---------------------------------------------

# input: some trajectories
# input: cluster group G1....Gn, where all trajectory in group Gi have dist<ε from each other
def trajStore_cluster(trajectories, epsilon=1.0):
    group = []
    trajectory_set = set()
    # 以 epsilon 为距离阈值 对轨迹进行聚类
    for trajectory in trajectories:
        if trajectory.traj_id in trajectory_set:
            continue
        cluster = [trajectory]
        trajectory_set.add(trajectory.traj_id)
        for other in trajectories:
            if other.traj_id in trajectory_set:
                continue
            # 如果二者距离满足阈值 则聚成一类
            if dist(trajectory, other) <= epsilon:
                cluster.append(other)
                trajectory_set.add(other.traj_id)
        group.append(cluster)
    # 对聚类后的轨迹簇 进行多轨迹的压缩  => 即将非参考轨迹的 (t,x,y) 转变为 (t,t')
    for cluster in group:
        # 如果该聚簇没有参考轨迹  原始存储
        if len(cluster) == 1:
            continue
        # 该聚簇有参考轨迹  选最长的作为参考轨迹 以获得最佳时间映射
        reference_trajectory = cluster[0]
        for i in range(len(cluster)):
            if cluster[i].traj_size > reference_trajectory.traj_size:
                reference_trajectory = cluster[i]
        # 对非参考轨迹进行压缩
        for i in range(len(cluster)):
            trajectory = cluster[i]
            if trajectory == reference_trajectory:
                continue
            trajectory.refe_traj_id = reference_trajectory.traj_id
            traj_dist(trajectory, reference_trajectory, True)
    return group


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


# 两点之间的距离  欧氏距离
def euc_dist(p1, p2):
    return round(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 5)


# 计算圆 与 直接相交的点
def line_intersect_circle(p, lsp, esp):
    # p is the circle parameter, lsp and lep is the two end of the line
    x0, y0, r0 = p
    x1, y1 = lsp
    x2, y2 = esp
    if r0 == 0:
        return [[x1, y1]]
    if abs(euc_dist([x0, y0], [x1, y1]) - r0) < 1e-5:
        return [[x1, y1]]
    if abs(euc_dist([x0, y0], [x2, y2]) - r0) < 1e-5:
        return [[x2, y2]]
    if x1 == x2:
        if abs(r0) >= abs(x1 - x0):
            p1 = x1, round(y0 - math.sqrt(r0 ** 2 - (x1 - x0) ** 2), 5)
            p2 = x1, round(y0 + math.sqrt(r0 ** 2 - (x1 - x0) ** 2), 5)
            inp = [p1, p2]
            # select the points lie on the line segment
            inp = [p for p in inp if p[0] >= min(x1, x2) and p[0] <= max(x1, x2)]
        else:
            inp = []
    else:
        k = (y1 - y2) / (x1 - x2)
        b0 = y1 - k * x1
        a = k ** 2 + 1
        b = 2 * k * (b0 - y0) - 2 * x0
        c = (b0 - y0) ** 2 + x0 ** 2 - r0 ** 2
        delta = b ** 2 - 4 * a * c
        if delta >= 0:
            p1x = round((-b - math.sqrt(delta)) / (2 * a), 5)
            p2x = round((-b + math.sqrt(delta)) / (2 * a), 5)
            p1y = round(k * p1x + b0, 5)
            p2y = round(k * p2x + b0, 5)
            inp = [[p1x, p1y], [p2x, p2y]]
            # select the points lie on the line segment
            inp = [p for p in inp if p[0] >= min(x1, x2) and p[0] <= max(x1, x2)]
        else:
            inp = []

    return inp if inp != [] else [[x1, y1]]


# 点与线段的交点  交点在线段上 则返回交点  不在线段上  则返回线段两段距离点最近的点
def point_intersect_line(p, l1, l2):
    x0, y0 = p
    x1, y1 = l1
    x2, y2 = l2
    k1 = (y2 - y1) / (x2 - x1)
    b1 = y1 - x1 * k1
    k0 = 1 / k1
    b0 = y0 - x0 * k0
    intersect_x = round((b1 - b0) / (k0 - k1), 5)
    if intersect_x > max(x1, x2) or intersect_x < min(x1, x2):
        dist_l1_to_p = euc_dist([x0, y0], [x1, y1])
        dist_l2_to_p = euc_dist([x0, y0], [x2, y2])
        if dist_l1_to_p < dist_l2_to_p:
            return [x1, y1]
        else:
            return [x2, y2]
    return [intersect_x, intersect_x * k0 + b0]


# 计算轨迹 t1 和 t2 之间的距离
# 距离 = max(t1各点 与 对应的 t2中的点 的欧式距离)
# pattern：False 时仅计算距离 True 时进行时间映射  t1的位置映射为t2的时间
def traj_dist(t1, t2, pattern=False):
    max_dist = -1
    # 1) 对于t1中的每个点 Pi 计算 t2中对应的位置 Pi'
    for i in range(len(t1.points)):
        print("i: ", i)
        if i == 163:
            print(111)
        t1_point = t1.points[i]
        # 1.1 计算 Pi 到第一个点的线性距离 distance_for_pi
        distance_for_pi = euc_dist(t1_point.to_list(), t1.points[0].to_list())

        print("t1_point:", t1_point.x, t1_point.y)
        print("distance_for_pi:", distance_for_pi)

        # 1.2 找到 t2 中距离第一个点 distance_for_pi 距离的位置  本质是一个圆与一个线段的交点
        # 隔 t1_point 最近的两个点 需要存储  t2中防止找不到对应点
        nc_point_closed_index = 0
        nc_point_closed_error = sys.maxsize
        # 正式开始
        index = 0
        index_set = set()
        while index < len(t2.points) - 1:
            if index == 162:
                print(111)
            pre_dist = euc_dist(t2.points[index].to_list(), t2.points[0].to_list())
            aft_dist = euc_dist(t2.points[index + 1].to_list(), t2.points[0].to_list())
            if abs((pre_dist + aft_dist) / 2 - distance_for_pi) < nc_point_closed_error:
                nc_point_closed_index = index
                nc_point_closed_error = abs((pre_dist + aft_dist) / 2 - distance_for_pi)
            # print("pre_dist:", pre_dist)
            # print("aft_dist:", aft_dist)
            # 线性距离在 两点之间  为了保证精度 每个对应点位置的确定都遍历一遍 t2
            if (pre_dist <= distance_for_pi <= aft_dist) or (pre_dist >= distance_for_pi >= aft_dist):
                index_set.add(index)
            index += 1
        # 将最后一个点 连接 起点 找可能存在的一个映射点 => 即伪造这么一个线段（假设它最后回到了起点） 加入lndex_set中 不会出现找不到映射点问题 解决 映射点精度过低问题
        # 此时 index=最后一个点
        pre_dist = euc_dist(t2.points[index].to_list(), t2.points[0].to_list())
        aft_dist = 0
        if pre_dist >= distance_for_pi >= aft_dist:
            index_set.add(index)

        # 2) 计算 Pi 与 Pi' 的欧氏距离
        # 如果 distance_for_pi 已经超出了 t2 的范围   即找不到对应点  选择nc_point_closed_index
        print(1111)
        if index == len(t2.points) - 1 and len(index_set) == 0:
            print("j:", index)
            line_pre = t2.points[nc_point_closed_index]
            line_next = t2.points[nc_point_closed_index + 1]
            nc_point = point_intersect_line(t1_point.to_list(), line_pre.to_list(), line_next.to_list())
            dist1 = get_haversine(nc_point, t1_point.to_list())
            dist2 = get_haversine(t1_point.to_list(), t2.points[index].to_list())
            # 如果非最佳匹配点 隔 t1_point 距离 小于 最后一个点 隔  t1_point 的距离 那就存前者
            if dist1 < dist2:
                max_dist = max(max_dist, dist1)
                delta_t = euc_dist(line_pre.to_list(), nc_point) / euc_dist(line_pre.to_list(),
                                                                            line_next.to_list()) * (
                                  line_next.t - line_pre.t)
                if pattern:
                    t1.refe_time.append(line_pre.t + int(delta_t))
            # 否则就存后者
            else:
                max_dist = max(max_dist, dist2)
                if pattern:
                    t1.refe_time.append(t2.points[index].t)
            if max_dist > 500:
                print(111111)
        else:
            # 范围内 找对应点
            circle_point = t2.points[0]
            best_closed_distance = sys.maxsize
            best_position = 0
            # 找出其中最接近原始位置的 映射点
            for position in index_set:
                line_pre = t2.points[position]
                line_next = t2.points[0]
                if position != index:
                    line_next = t2.points[position + 1]
                corresponding_point = \
                    line_intersect_circle((circle_point.x, circle_point.y, distance_for_pi), (line_pre.x, line_pre.y),
                                          (line_next.x, line_next.y))
                corresponding_point = corresponding_point[0]
                if get_haversine(corresponding_point, t1_point.to_list()) < best_closed_distance:
                    best_position = position
                    best_closed_distance = get_haversine(corresponding_point, t1_point.to_list())

            # 计算与原始位置的 地理距离
            print("j: ", best_position)
            line_pre = t2.points[best_position]
            line_next = t2.points[0]
            if best_position != index:
                line_next = t2.points[best_position + 1]
            corresponding_point = \
                line_intersect_circle((circle_point.x, circle_point.y, distance_for_pi), (line_pre.x, line_pre.y),
                                      (line_next.x, line_next.y))
            # 如果 pattern = true 说明此时需要进行时间匹配
            corresponding_point = corresponding_point[0]

            # 如果 非对应点 比 对应点 的距离还进  必然选择非对应点
            nc_point = point_intersect_line(t1_point.to_list(), t2.points[nc_point_closed_index].to_list(),
                                            t2.points[nc_point_closed_index + 1].to_list())
            if get_haversine(nc_point, t1_point.to_list()) < get_haversine(corresponding_point, t1_point.to_list()):
                line_pre = t2.points[nc_point_closed_index]
                line_next = t2.points[nc_point_closed_index + 1]
                corresponding_point = nc_point

            print("corresponding_point:", corresponding_point)
            if pattern:
                # s1/s=t1/t => t1=(s1/s)*t
                delta_t = euc_dist(line_pre.to_list(), corresponding_point) / euc_dist(line_pre.to_list(),
                                                                                       line_next.to_list()) * (
                                  line_next.t - line_pre.t)
                t1.refe_time.append(line_pre.t + int(delta_t))
            max_dist = max(get_haversine(t1_point.to_list(), corresponding_point), max_dist)
            if max_dist > 500:
                print(111111)
        print("max_dist:", max_dist)
    # 3）所有点中最大的距离 即为 return 值
    return max_dist


# 计算轨迹间的距离 return max(traj_dist(t1,t2),traj_dist(t2,t1))
def dist(t1, t2):
    return max(traj_dist(t2, t1), traj_dist(t1, t2))


if __name__ == '__main__':
    print(get_haversine([29.83, 121.228], [29.82, 121.229]))
    # 大概实际距离是 欧式距离的10000倍
    # 1116.1660304832283
    # 0.01005
    # print(euc_dist([30.1517,120.2117], [40.0643, 116.5987]))
    # print(euc_dist([30.1575,120.2254], [40.0643, 116.5987]))
    # line_pre = [30.1517,120.2117]
    # line_next = [30.1575,120.2254]
    # corresponding_point = \
    #     line_intersect_circle((40.0643, 116.5987, 10.54859), (30.1575,120.2254),
    #                           (40.0643, 116.5987))
    # corresponding_point = corresponding_point[0]
