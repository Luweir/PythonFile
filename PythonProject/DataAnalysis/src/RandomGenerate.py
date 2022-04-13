import scipy.stats as stats
import numpy as np


# 根据均值和标准差生成随机数据
def generate_random(mean, std, min, max, count):
    a, b = min, max
    mu, sigma = mean, std
    dist = stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)

    values = dist.rvs(count)
    return values
