# Copyright (c) 2022 Dai HBG


"""
该文档定义用于常见计算的脚本
"""


import numpy as np
from libc.math cimport isnan, sqrt


def div_2d(double[:, :] a, double[:, :] b):  # 矩阵除法，重要的是将不合法的地方置为nan
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if (isnan(a[i, j])) or isnan(b[i, j]) or (b[i, j] == 0):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] / b[i, j]
    return s


def div_2d_32(float[:, :] a, float[:, :] b):  # 矩阵除法，重要的是将不合法的地方置为nan
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2), dtype=np.float32)
    cdef float[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if (isnan(a[i, j])) or isnan(b[i, j]) or (b[i, j] == 0):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] / b[i, j]
    return s