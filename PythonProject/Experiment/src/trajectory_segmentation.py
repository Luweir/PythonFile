import math
from dateutil import parser

import numpy as np
import pandas as pd


# from geopy.distance import geodesic


# 得到两点距离
# print(geodesic((31.6574, 119.0347), (32.3828, 119.4044)).m)  # 最快
# print(haversine(119.0347, 31.6574, 119.4044, 32.3828)) # 最准
def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000


# 依据特征对轨迹进行分割

if __name__ == '__main__':
    filename = "new_10.9.xlsx"
    points = pd.read_excel("../data/" + filename)
    points = points.values.tolist()
    speed = [0]
    for i in range(1, len(points)):
        dis_time = (parser.parse(points[i][0]) - parser.parse(points[i - 1][0])).seconds
        distance = haversine(points[i][2], points[i][1], points[i - 1][2], points[i - 1][1])
        speed.append(round(distance / dis_time, 4))
    print(speed)
