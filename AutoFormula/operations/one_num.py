# Copyright (c) 2022 Dai HBG


"""
numpy version
"""

import cupy as cp
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def powv(a: np.array, num: float):  # 幂函数运算符
    s = np.zeros(a.shape, dtype=np.float32)
    s[(a > 0) & (~np.isnan(a))] = a[(a > 0) & (~np.isnan(a))] ** num
    s[(a < 0) & (~np.isnan(a))] = -((-a[(a < 0) & (~np.isnan(a))]) ** num)
    s[~np.isnan(a)] = np.nan
    return s


def tsmax(a, num: int):  # 计算时序非nan最大值
    s = np.zeros(a.shape, dtype=np.float32)
    if len(a.shape) == 2:
        ss = np.lib.stride_tricks.as_strided(a, shape=(a.shape[0] - num + 1, num, a.shape[1]),
                                             strides=a.itemsize * np.array([a.shape[1], a.shape[1], 1]))
    else:
        ss = np.lib.stride_tricks.as_strided(a, shape=(a.shape[0] - num + 1, num, a.shape[1], a.shape[2]),
                                             strides=a.itemsize * np.array([a.shape[1] * a.shape[2],
                                                                            a.shape[1] * a.shape[2],
                                                                            a.shape[2], 1]))
    is_nan = np.sum(~np.isnan(ss), axis=1) == 0
    s[:num - 1] = np.nan
    tmp = np.nanmax(ss, axis=1)
    tmp[is_nan] = np.nan
    s[num - 1:] = tmp
    return s


def tsmaxpos(a: np.array, num: int):  # 返回最大值位置
    s = a.copy()
    s[~np.isnan(s)] = np.nanmin(s) - 100
    s_min = np.min(s)
    if len(a.shape) == 2:
        ss = np.lib.stride_tricks.as_strided(a, shape=(a.shape[0] - num + 1, num, a.shape[1]),
                                             strides=a.itemsize * np.array([a.shape[1], a.shape[1], 1]))
    else:
        ss = np.lib.stride_tricks.as_strided(a, shape=(a.shape[0] - num + 1, num, a.shape[1], a.shape[2]),
                                             strides=a.itemsize * np.array([a.shape[1] * a.shape[2],
                                                                            a.shape[1] * a.shape[2],
                                                                            a.shape[2], 1]))
    s[:num - 1] = np.nan
    tmp = np.argmax(ss, axis=1)
    tmp_max = np.max(ss, axis=1)
    tmp[tmp_max == s_min] = np.nan
    s[num - 1:] = tmp
    return s
