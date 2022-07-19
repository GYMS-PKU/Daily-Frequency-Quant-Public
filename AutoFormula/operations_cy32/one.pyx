# Copyright (c) 2022 Dai HBG


"""
one input
"""


import numpy as np
from libc.math cimport isnan, log, exp


def neg_2d(float[:, :] a):  # 二维矩阵求相反数
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    s = np.zeros((dim_1, dim_2), dtype=np.float32)
    cdef float[:, :] s_view = s
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = -a[i, j]
    return s


def neg_3d(float[:, :, :] a):  # 三维矩阵求相反数
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k
    s = np.zeros((dim_1, dim_2, dim_3), dtype=np.float32)
    cdef float[:, :, :] s_view = s
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = -a[i, j, k]

    return s


def absv_2d(float[:, :] a):  # 二维矩阵取绝对值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    s = np.zeros((dim_1, dim_2), dtype=np.float32)
    cdef float[:, :] s_view = s
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            if a[i, j] < 0:
                s_view[i, j] = -a[i, j]
            else:
                s_view[i, j] = a[i, j]
    return s


def absv_3d(float[:, :, :] a):  # 三维矩阵取绝对值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k
    s = np.zeros((dim_1, dim_2, dim_3), dtype=np.float32)
    cdef float[:, :, :] s_view = s
    cdef nan = np.nan
    for i in range(dim_1):
       for j in range(dim_2):
           for k in range(dim_3):
               if isnan(a[i, j, k]):
                   s_view[i, j, k] = nan
                   continue
               if a[i, j, k] < 0:
                   s_view[i, j, k] = -a[i, j, k]
               else:
                   s_view[i, j, k] = a[i, j, k]
    return s


def logv_2d(float[:, :] a):  # 二维矩阵求log
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    s = np.zeros((dim_1, dim_2), dtype=np.float32)
    cdef float[:, :] s_view = s
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            if a[i, j] > 0:
                s_view[i, j] = log(a[i, j] + 1e-7)
            else:
                s_view[i, j] = nan
    return s


def logv_3d(float[:, :, :] a):  # 三维矩阵求log
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k
    s = np.zeros((dim_1, dim_2, dim_3), dtype=np.float32)
    cdef float[:, :, :] s_view = s
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                if a[i, j, k] > 0:
                    s_view[i, j, k] = log(a[i, j, k] + 1e-7)
                else:
                    s_view[i, j, k] = nan
    return s


def expv_2d(float[:, :] a):  # 二维矩阵求log
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    s = np.zeros((dim_1, dim_2), dtype=np.float32)
    cdef float[:, :] s_view = s
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = exp(a[i, j])
    return s


def expv_3d(float[:, :, :] a):  # 三维矩阵求指数
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k
    s = np.zeros((dim_1, dim_2, dim_3), dtype=np.float32)
    cdef float[:, :, :] s_view = s
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = exp(a[i, j, k])
    return s


def intracumsum_3d(float[:, :, :] a):  # 日内维度求累积和
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k
    s = np.zeros((dim_1, dim_2, dim_3), dtype=np.float32)
    cdef float[:, :, :] s_view = s
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if j == 0:
                    s_view[i, j, k] = a[i, j, k]
                    continue
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = s_view[i, j - 1, k]
                    continue
                if isnan(s_view[i, j - 1, k]):
                    s_view[i, j, k] = a[i, j, k]
                    continue
                s_view[i, j, k] = s_view[i, j - 1, k] + a[i, j, k]
    return s
