# Copyright (c) 2022 Dai HBG


"""
本脚本对收益率序列进行风险度量
2022-07-19
- need to modify
"""


import numpy as np


def risk_measure(ret: np.array) -> tuple:
    max_d = 0  # 最大回撤
    max_loss_day = 0  # 最大亏损天数
    cul_ret = 0  # 累计盈亏

    last_high = 0  # 前期高点
    last_low = 0

    last_high_pos = 0
    last_low_pos = 0

    start = 0
    end = 0

    i = 0
    while i < len(ret):
        cul_ret += ret[i]
        if cul_ret > last_high:  # 创新高
            last_high = cul_ret
            last_high_pos = i
            last_low = cul_ret
            last_low_pos = i
        elif cul_ret < last_low:  # 创新低
            last_low = cul_ret
            last_low_pos = i
            if last_high - last_low > max_d:
                max_d = last_high - last_low  # 更新最大回撤
        else:
            if i - last_high_pos > max_loss_day:
                max_loss_day = i - last_high_pos
                start = last_high_pos
                end = i
        i += 1

    return max_d, max_loss_day, start, end


