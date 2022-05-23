# -*- coding: utf-8 -*-
import math
import random

import numpy as np
import sys
import pickle

import importlib
def euc_dist(p1, p2):
    return round(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 5)
if __name__ == '__main__':
    print(euc_dist([40.0638,116.599],[40.0119,116.6055]))
