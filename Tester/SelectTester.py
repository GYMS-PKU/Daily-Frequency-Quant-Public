# Copyright (c) 2022 Dai HBG

"""
定义根据选中信号选中股票回测
2022-06-28
- 更新：新增对可转债的支持
"""

import numpy as np


class TesterStats:
    def __init__(self, log_dic: dict, mean_ret: float, sharpe_ratio: float, win_rate: float, ret:np.array):
        self.log_dic = log_dic
        self.mean_ret = mean_ret
        self.sharpe_ratio = sharpe_ratio
        self.win_rate = win_rate
        self.ret = ret


class SelectTester:
    def __init__(self):
        pass

    @staticmethod
    def test(signal: np.array, ret: np.array, top: np.array, zdt_top: np.array, position_date_dic: dict,
             order_code_dic: dict, s: int, e: int):
        abs_ret = []  # 绝对收益
        # alpha_ret = []  # 超额收益

        log_dic = {}  # 交易日志
        stocks = np.array([order_code_dic[i] for i in range(len(order_code_dic))])
        for i in range(s, e):
            date = position_date_dic[i]
            pos_length = np.sum(top[i] & (~np.isnan(signal[i])) & (signal[i] > 0))
            # pos = top[i] & (~np.isnan(signal[i])) & (signal[i] > 0)
            se = top[i] & (~np.isnan(signal[i])) & (signal[i] > 0) & zdt_top[i]
            if np.sum(se) == 0:
                log_dic[date] = {'stocks': [], 'ret': [],
                                 'sig': []}
                abs_ret.append(0)
                continue
            tmp_ret = ret[i, se]
            tmp_ret[np.isnan(tmp_ret)] = 0
            abs_ret.append(np.sum(tmp_ret-0.0015) / pos_length)
            log_dic[date] = {'stocks': stocks[se], 'ret': tmp_ret,
                             'sig': signal[i, se]}
        abs_ret = np.array(abs_ret)
        stats = TesterStats(log_dic=log_dic, mean_ret=float(np.mean(abs_ret)),
                            sharpe_ratio=np.mean(abs_ret) / np.std(abs_ret),
                            win_rate=np.sum(abs_ret > 0) / np.sum(abs_ret != 0) if np.sum(abs_ret != 0) > 0 else 0,
                            ret=abs_ret)
        return stats
