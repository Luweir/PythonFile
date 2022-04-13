import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser
import os
import datetime

if __name__ == '__main__':
    # 一、日期操作
    # 1.1 字符串转日期
    date_str1 = "2021/10/11 4:55:10"
    # datetime.datetime.strptime 按格式解析
    date_object = datetime.datetime.strptime(date_str1, '%Y/%m/%d %H:%M:%S')
    print(date_object)
    # parser.parse 自动解析
    date_object = parser.parse(date_str1)
    print(date_object)
    # 1.2 时间转字符串
    print(type(str(date_object)))

    # 1.3 日期相加减，只能用增量加减
    new_date_object = date_object + datetime.timedelta(hours=+0.5)  # days、hours ......
    print("相加前：" + str(date_object))
    print("相加后：" + str(new_date_object))

    # 1.4 比较日期大小
    print(new_date_object > date_object)
    print(((date_object + datetime.timedelta(seconds=+30)) - date_object).seconds)

    # 二、时间操作
    time_str1 = "4:55:10"
    time_str2 = "4:55:30"

    new_time = datetime.datetime.strptime(time_str2, "%H:%M:%S") + datetime.timedelta(seconds=+40)
    print(new_time.time())
