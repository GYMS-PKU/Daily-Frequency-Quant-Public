# Copyright (c) 2021 Dai HBG


"""
two input
"""


import numpy as np
from libc.math cimport isnan


def add_2d(float[:, :] a, float[:, :] b):  # 二维矩阵加法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2), dtype=np.float32)
    cdef float[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if (isnan(a[i, j])) or isnan(b[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] + b[i, j]
    return s


def add_num_2d(float[:, :] a, float b):  # 二维矩阵数字加法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2), dtype=np.float32)
    cdef float[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] + b
    return s


def add_3d(float[:, :, :] a, float[:, :, :] b):  # 三维矩阵加法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3), dtype=np.float32)
    cdef float[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i, j, k])) or isnan(b[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] + b[i, j, k]
    return s


def add_num_3d(float[:, :, :] a, float b):  # 三维矩阵数字加法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3), dtype=np.float32)
    cdef float[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] + b
    return s


def minus_2d(float[:, :] a, float[:, :] b):  # 二维矩阵减法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2), dtype=np.float32)
    cdef float[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if (isnan(a[i, j])) or isnan(b[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] - b[i, j]
    return s


def minus_num_2d(float[:, :] a, float b):  # 二维矩阵数字减法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2), dtype=np.float32)
    cdef float[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] - b
    return s


def minus_3d(float[:, :, :] a, float[:, :, :] b):  # 三维矩阵减法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3), dtype=np.float32)
    cdef float[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i, j, k])) or isnan(b[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] - b[i, j, k]
    return s


def minus_num_3d(float[:, :, :] a, float b):  # 三维矩阵数字减法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3), dtype=np.float32)
    cdef float[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] - b
    return s


def prod_2d(float[:, :] a, float[:, :] b):  # 二维矩阵乘法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2), dtype=np.float32)
    cdef float[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if (isnan(a[i, j])) or isnan(b[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] * b[i, j]
    return s


def prod_num_2d(float[:, :] a, float b):  # 二维矩阵数字乘法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2), dtype=np.float32)
    cdef float[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] * b
    return s


def prod_3d(float[:, :, :] a, float[:, :, :] b):  # 三维矩阵乘法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3), dtype=np.float32)
    cdef float[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i, j, k])) or isnan(b[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] * b[i, j, k]
    return s


def prod_num_3d(float[:, :, :] a, float b):  # 三维矩阵数字乘法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3), dtype=np.float32)
    cdef float[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] * b
    return s


def div_2d(float[:, :] a, float[:, :] b):  # 二维矩阵除法
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


def div_num_2d(float[:, :] a, float b):  # 二维矩阵数字除法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2), dtype=np.float32)
    cdef float[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]) or (b == 0):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] / b
    return s


def div_3d(float[:, :, :] a, float[:, :, :] b):  # 三维矩阵除法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3), dtype=np.float32)
    cdef float[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i, j, k])) or isnan(b[i, j, k]) or (b[i, j, k] == 0):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] / b[i, j, k]
    return s


def div_num_3d(float[:, :, :] a, float b):  # 三维矩阵数字除法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3), dtype=np.float32)
    cdef float[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]) or (b == 0):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] / b
    return s
