# Copyright (c) 2021-2022 Dai HBG


"""
numpy version
"""


import numpy as np


def neg(a):
    s = a.copy()
    s[~np.isnan(a)] = - a[~np.isnan(a)]
    return s


def absv(a):  # 取绝对值
    a[~np.isnan(a)] = np.abs(a[~np.isnan(a)])
    return a


def logv(a):
    a[(~np.isnan(a)) & (a > 0)] = np.log(a[(~np.isnan(a)) & (a > 0)] + 1e-7)
    a[(~np.isnan(a)) & (a <= 0)] = np.nan
    return a
