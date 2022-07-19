# Copyright (c) 2022 Dai HBG


"""
numpy version
"""


import numpy as np
import warnings
warnings.filterwarnings("ignore")


def add(a, b):
    return a + b


def minus(a, b):
    return a - b


def prod(a, b):
    return a * b


def div(a, b):
    s = a / b
    s[np.isinf(s)] = np.nan
    return s


def intratsregres(a: np.array, b: np.array):  # 日内时序回归残差，有一个缺失值就置为缺失值
    beta = np.nanmean(a * b, axis=1)
    beta -= np.nanmean(a, axis=1) * np.nanmean(b, axis=1)

    beta /= np.var(a, axis=1)

    beta[np.isinf(beta)] = 0
    s = b.transpose(1, 0, 2) - np.nanmean(b, axis=1)

    s -= beta * (a.transpose(1, 0, 2) - np.nanmean(a, axis=1))

    return s.transpose(1, 0, 2)


def intratsregbeta(a: np.array, b: np.array):  # 日内回归alpha
    beta = np.nanmean(a * b, axis=1)
    beta -= np.nanmean(a, axis=1) * np.nanmean(b, axis=1)

    beta /= np.var(a, axis=1)

    beta[np.isinf(beta)] = 0
    return beta


def intratsregalpha(a: np.array, b: np.array):  # 日内回归alpha
    beta = np.nanmean(a * b, axis=1)
    beta -= np.nanmean(a, axis=1) * np.nanmean(b, axis=1)

    beta /= np.var(a, axis=1)

    beta[np.isinf(beta)] = 0
    s = np.nanmean(b, axis=1)

    s -= beta * np.nanmean(a, axis=1)

    return s


def lt(a, b):
    s = (a < b).astype(np.float32)
    s[np.isnan(a)] = np.nan
    s[np.isnan(b)] = np.nan
    return s


def le(a, b):
    s = (a <= b).astype(np.float32)
    s[np.isnan(a)] = np.nan
    s[np.isnan(b)] = np.nan
    return s


def gt(a, b):
    s = (a > b).astype(np.float32)
    s[np.isnan(a)] = np.nan
    s[np.isnan(b)] = np.nan
    return s


def ge(a, b):
    s = (a >= b).astype(np.float32)
    s[np.isnan(a)] = np.nan
    s[np.isnan(b)] = np.nan
    return s
