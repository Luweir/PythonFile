import Experiment.FastDTW.fast_dtw_neigbors as dtw
import numpy as np
import math


## DTW Distance

def d(x, y):
    return np.sum((x - y) ** 2)


def euc_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


EARTH_RADIUS = 6371229  # m 用于两点间距离计算


# 对的
def get_haversine(lat1, lat2, lon1, lon2):
    lat1 = lat1 * math.pi / 180
    lat2 = lat2 * math.pi / 180
    lon1 = lon1 * math.pi / 180
    lon2 = lon2 * math.pi / 180
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a_a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a_a), math.sqrt(1 - a_a))
    return EARTH_RADIUS * c

# 多维DTW  xy代表经纬度
def dtw_distance1(s1, s2, mww=10000):
    ts_a = np.array(s1)
    ts_b = np.array(s2)
    M, N = np.shape(ts_a)[0], np.shape(ts_b)[0]

    cost = np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = euc_dist(ts_a[0], ts_b[0])
    print(cost[0, 0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + euc_dist(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + euc_dist(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + euc_dist(ts_a[i], ts_b[j])
    print(cost)
    # Return DTW distance given window
    return cost[-1, -1]


def dtw_distance2(ts_a, ts_b, d=lambda x, y: abs(x - y), mww=10000):
    """Computes dtw distance between two time series

    Args:
        ts_a: time series a
        ts_b: time series b
        d: distance function
        mww: max warping window, int, optional (default = infinity)

    Returns:
        dtw distance
    """

    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = np.shape(ts_a)[1], np.shape(ts_b)[1]
    cost = np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    print(cost[0, 0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]


if __name__ == '__main__':
    print(2 ** 3)
    series1 = [0, 1, 3, 5, 7, 9]
    series2 = [0, 2, 4, 6, 8, 10]
    print(dtw_distance1([[0, 0], [2, 2], [3, 3], [6, 5], [6, 6], [8, 8], [10, 10]],
                        [[0, 0], [2, 2], [4, 4], [6, 6], [8, 8], [10, 10]]))
    print(dtw.get_fastdtw([[0, 0], [2, 2], [3, 3], [6, 5], [6, 6], [8, 8], [10, 10]],
                          [[0, 0], [2, 2], [4, 4], [6, 6], [8, 8], [10, 10]]))
    s1 = np.array([[0, 0], [1, 1]])
    s2 = np.array([[3, 3], [2, 2]])
    # print(d(s1,s2))
