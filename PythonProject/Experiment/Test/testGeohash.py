import math
import scipy.stats as stats
import numpy as np
import Experiment.FastDTW.fast_dtw_neigbors as dtw
import Experiment.Test.geo_hash as gh
import pandas as pd

if __name__ == '__main__':
    data = pd.read_excel("../data/10.9.xlsx", usecols=[1, 2])
    data = np.array(data)
    latitude = np.array(data[:, 0])
    longitude = np.array(data[:, 1])
    latitude_interval = [-90.0, -45.0, 0.0, 45.0, 90.0]
    longitude_interval = [-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0]
    latitude_min = math.floor(latitude.min())
    latitude_max = math.ceil(latitude.max())
    longitude_min = math.floor(longitude.min())
    longitude_max = math.ceil(longitude.max())

    res = gh.encode(40.0646, 116.5987, latitude_min, latitude_max, longitude_min, longitude_max, precision=6)
    print('Geohash for 40.0646,116.5987:', res)
    print('Exact coordinate for Geohash ' + res + ' :\n',
          gh.decode_exactly(res, latitude_min, latitude_max, longitude_min, longitude_max))

    res = gh.encode(40.0117, 116.6064, latitude_min, latitude_max, longitude_min, longitude_max, precision=6)
    print('Geohash for 40.0117,116.6064:', res)
    print('Exact coordinate for Geohash ' + res + ' :\n',
          gh.decode_exactly(res, latitude_min, latitude_max, longitude_min, longitude_max))

    # res = gh.encode(38.371, 116.8924, latitude_interval[math.floor(latitude.min() / 45) + 2],
    #                 latitude_interval[math.ceil(latitude.max() / 45) + 2],
    #                 longitude_interval[math.floor(longitude.min() / 45) + 4],
    #                 longitude_interval[math.ceil(longitude.max() / 45) + 4],
    #                 precision=6)
    # print('Geohash for 38.371, 116.8924:', res)
    # print('Exact coordinate for Geohash ' + res + ' :\n',
    #       gh.decode_exactly(res, latitude_interval[math.floor(latitude.min() / 45) + 2],
    #                         latitude_interval[math.ceil(latitude.max() / 45) + 2],
    #                         longitude_interval[math.floor(longitude.min() / 45) + 4],
    #                         longitude_interval[math.ceil(longitude.max() / 45) + 4]))
