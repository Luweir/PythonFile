import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser
import os
import datetime


# 将xlsx和plt格式数据转换为csv格式
def to_csv(filename):
    name, suffix = os.path.splitext(filename)
    data = pd.DataFrame()
    if suffix == ".xlsx":
        data = pd.read_excel("../data/" + filename, header=0, names=['time', 'latitude', 'longitude'])
        timestamp = []
        for ele in data['time']:
            timestamp.append(date_to_timestamp(ele))
        data['time'] = timestamp
    if suffix == ".csv" or suffix == ".plt":
        data = pd.read_csv("../data/" + filename, header=None, sep=' ')
    pd.DataFrame(data).to_csv("../data/" + name + ".csv", index=False, header=0, sep=',')


def date_to_timestamp(date):
    return int(time.mktime(time.strptime(str(date), "%Y/%m/%d %H:%M:%S")))


if __name__ == '__main__':
    to_csv("10.12.xlsx")
