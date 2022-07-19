# Copyright (c) 2022 Dai HBG


"""
该脚本用于测试某一个信号矩阵top n股票做多的收益

日志
2022-01-21
- 需要过滤涨停板
"""


import numpy as np


def top_n_tester(signal: np.array, ret: np.array, top: np.array, zdt_top: np.array, position_date_dic: dict,
                 order_code_dic: dict, s: int, e: int, n: int = 10):
    abs_ret = []  # 绝对收益
    alpha_ret = []  # 超额收益

    log_dic = {}  # 交易日志
    stocks = np.array([order_code_dic[i] for i in range(len(order_code_dic))])
    for i in range(s, e):
        date = position_date_dic[i]
        se = top[i] & (~np.isnan(signal[i])) & zdt_top[i]
        arg_sig = np.argsort(signal[i, se])
        abs_ret.append(np.nanmean(ret[i, se][arg_sig[-n:]]))
        alpha_ret.append(np.nanmean(ret[i, se][arg_sig[-n:]]) - np.nanmean(ret[i, se]))
        log_dic[date] = {'stocks': stocks[se][arg_sig[-n:]], 'ret': ret[i, se][arg_sig[-n:]],
                         'sig': signal[i, se][arg_sig[-n:]]}
    return log_dic, abs_ret, alpha_ret
