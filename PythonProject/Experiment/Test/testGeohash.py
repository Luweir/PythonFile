import math
import numpy as np
import Experiment.Test.geo_hash as gh
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("../data/10.9.csv", usecols=[0, 1, 2])
    data = np.array(data)
    latitude = np.array(data[:, 1])
    longitude = np.array(data[:, 2])
    # latitude_interval = [-90.0, -45.0, 0.0, 45.0, 90.0]
    # longitude_interval = [-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0]
    latitude_min = math.floor(latitude.min())
    latitude_max = math.ceil(latitude.max())
    longitude_min = math.floor(longitude.min())
    longitude_max = math.ceil(longitude.max())

    newData = []
    for i in range(len(data)):
        res = gh.encode(data[i, 1], data[i, 2], latitude_min, latitude_max, longitude_min, longitude_max, precision=6)
        newData.append([data[i, 0], res])
    # print(newData)
    for i in range(len(newData) - 1):
        cur = len(newData) - i - 1
        newData[cur][0] = newData[cur][0] - newData[cur - 1][0]
    pd.DataFrame(newData).to_csv("compressed_10.9.csv", columns=None, index=None)
    # res = gh.encode(40.0646, 116.5987, latitude_min, latitude_max, longitude_min, longitude_max, precision=6)
    # print('Geohash for 40.0646,116.5987:', res)
    # print('Exact coordinate for Geohash ' + res + ' :\n',
    #       gh.decode_exactly(res, latitude_min, latitude_max, longitude_min, longitude_max))
    #
    # res = gh.encode(40.0117, 116.6064, latitude_min, latitude_max, longitude_min, longitude_max, precision=6)
    # print('Geohash for 40.0117,116.6064:', res)
    # print('Exact coordinate for Geohash ' + res + ' :\n',
    #       gh.decode_exactly(res, latitude_min, latitude_max, longitude_min, longitude_max))
