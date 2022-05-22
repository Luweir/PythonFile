import numpy as np
import pandas as pd


# ---------------------------------------------
# author: luweir
# target: point class implement of trajectory
# date: 2022-5-13
# ---------------------------------------------
class Point:
    def __init__(self, x, y, t=None):
        self.t = t
        self.x = x
        self.y = y
        self.t2 = None  # 匹配点 t2 指向一个 Point

    # 转换成列表 不含t
    def to_list(self):
        return [self.x, self.y]

    # 含时间 t 转换为列表
    def to_list_t(self):
        return [self.t, self.x, self.y]
