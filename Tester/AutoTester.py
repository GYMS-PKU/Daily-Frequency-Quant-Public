# Copyright (c) 2021-2022 Dai HBG

"""
该代码定义的类用于计算一个信号的平均IC等统计值

日志
2021-08-30
- 定义：计算平均IC，信号自相关系数，IC_IR，IC为正的频率
2021-09-08
- 新增：统计信号排名最高的1个，5个，10个股票的平均收益，以评估信号的纯多头表现
2021-10-09
- 更新：统计平均收益应该也包含超额收益
2021-10-10
- 更新：stats应该存下top前50的所有超额收益率序列，因此可以后期选择需要的排位的股票
2021-10-15
- 更新：对于周频测试，可以使用字段fix_weekday指定计算周几的平均IC
2022-01-10
- 更新：新增不同的IC评价方式
- 更新：涨跌停判定需要另外传入一个判定矩阵，用来判定是否过滤
"""

import numpy as np


class IcCalculator:
    def __init__(self):
        pass

    def cal(self, signal: np.array, ret: np.array, top: np.array, method: str = 'IC', zdt: bool = False,
            zdt_top: np.array = None, param: dict = None) -> np.array:
        """
        :param signal: 信号矩阵
        :param ret: 收益率矩阵
        :param top: 合法矩阵
        :param method: 计算方式
        :param zdt: 是否过滤涨跌停
        :param zdt_top: 涨跌停收益率矩阵
        :param param: 参数
        :return:
        """
        ics = np.zeros(len(signal))
        for i in range(len(signal)):
            sig = signal[i, top[i]]
            r = ret[i, top[i]]
            if np.sum(~np.isnan(sig)) < 2 or np.sum(~np.isnan(r)) < 2:
                continue
            if method == 'IC':  # 普通计算截面IC
                se = (~np.isnan(sig)) & (~np.isnan(r))  # 选出都没有缺失值的
                if zdt:
                    se = se & (zdt_top[i, top[i]])  # 过滤涨跌停
                if np.sum(se) >= 2:
                    ics[i] = np.corrcoef(sig[se], r[se])[0, 1]
                else:
                    ics[i] = 0
            elif method == 'long_IC':  # 计算多头IC
                sig -= np.nanmean(sig)
                r -= np.nanmean(r)
                sig[np.isnan(sig)] = 0
                r[np.isnan(r)] = 0
                if i > 0:
                    se = abs(ret[i - 1, top[i]]) < 0.099  # 过滤涨跌停
                    if np.sum(se & (sig > 0)) >= 2:
                        cov = np.sum(sig[se & (sig > 0)] * r[se & (sig > 0)])
                        ics[i] = cov / (np.std(r[se]) * np.std(sig[se]))
                    else:
                        ics[i] = 0
                else:
                    cov = np.sum(sig[sig > 0] * r[sig > 0])
                    ics[i] = cov / (np.std(r) * np.std(sig))
            elif method == 'poly_IC':  # 给不同部分的股票给予不同权重
                if param is None:
                    degree = 2
                    print('no param, set degree to 2')
                else:
                    try:
                        degree = param['degree']
                    except KeyError:
                        degree = 2
                        print('no key \'degree\', set degree to 2')
                sig -= np.nanmean(sig)
                r -= np.nanmean(r)
                sig[np.isnan(sig)] = 0
                r[np.isnan(r)] = 0
                sig[sig > 0] = sig[sig > 0] ** degree
                sig[sig < 0] = -(-sig[sig < 0]) ** degree
                if i > 0:
                    se = abs(ret[i - 1, top[i]]) < 0.099  # 过滤涨跌停
                    if np.sum(se) >= 2:
                        ics[i] = np.corrcoef(sig[se], r[se])[0, 1]
                    else:
                        ics[i] = 0
                else:
                    ics[i] = np.corrcoef(sig, r)[0, 1]
            elif method == 'long_poly_IC':
                if param is None:
                    degree = 2
                    print('no param, set degree to 2')
                else:
                    try:
                        degree = param['degree']
                    except KeyError:
                        degree = 2
                        print('no key \'degree\', set degree to 2')
                sig -= np.nanmean(sig)
                r -= np.nanmean(r)
                sig[np.isnan(sig)] = 0
                r[np.isnan(r)] = 0
                sig[sig > 0] = sig[sig > 0] ** degree
                sig[sig < 0] = -(-sig[sig < 0]) ** degree
                if i > 0:
                    se = abs(ret[i - 1, top[i]]) < 0.099  # 过滤涨跌停
                    if np.sum(se & (sig > 0)) >= 2:
                        cov = np.sum(sig[se & (sig > 0)] * r[se & (sig > 0)])
                        ics[i] = cov / (np.std(r[se]) * np.std(sig[se]))
                    else:
                        ics[i] = 0
                else:
                    cov = np.sum(sig[sig > 0] * r[sig > 0])
                    ics[i] = cov / (np.std(r) * np.std(sig))
            elif 'long_top_' in method:  # 计算多头靠前的部分
                se = (~np.isnan(sig)) & (~np.isnan(r))  # 选出都没有缺失值的
                n = int(method.split('_')[-1])
                # sig[np.isnan(sig)] = 0
                r -= np.nanmean(r)
                # r[np.isnan(r)] = 0
                if i > 0:
                    se = se & (abs(ret[i - 1, top[i]]) < 0.099)  # 过滤涨跌停
                    if np.sum(se) >= 2:
                        arg_sig = np.argsort(sig[se])
                        ics[i] = np.mean(r[se][arg_sig[-n:]])
                    else:
                        ics[i] = 0
                else:
                    arg_sig = np.argsort(sig[se])
                    ics[i] = np.mean(r[se][arg_sig[-n:]])
        return ics


class Stats:
    def __init__(self):
        self.ICs = []
        self.mean_IC = 0
        # self.auto_corr = 0
        self.IC_IR = 0
        self.positive_IC_rate = 0
        # self.top_n_ret = {i - 1: [] for i in range(1, 51)}  # 存储多头超额收益
        # self.top_n_raw_ret = {i - 1: [] for i in range(1, 51)}  # 存储多头收益


class AutoTester:
    def __init__(self):
        self.icc = IcCalculator()

    def test(self, signal: np.array, ret: np.array, top: np.array = None, method: str = 'IC',
             param: dict = None, zdt: bool = True, zdt_top: np.array = None) -> Stats:
        """
        :param signal: 信号矩阵
        :param ret: 和信号矩阵形状一致的收益率矩阵，意味着同一个时间维度已经做了delay
        :param top: 每个时间截面上进入截面的股票位置
        :param zdt: 是否过滤zdt
        :param zdt_top: zdt对应的收益率矩阵
        :return: 返回Stats类的实例
        """
        if top is None:
            top = signal != 0
        if zdt_top is None:
            zdt_top = np.zeros(top.shape)
            zdt_top[:] = True
        ics = []
        auto_corr = []
        assert len(signal) == len(ret)
        assert len(signal) == len(top)
        stats = Stats()
        ics = self.icc.cal(signal=signal, ret=ret, top=top, method=method, param=param, zdt=zdt, zdt_top=zdt_top)
        # print(ics)
        # ics = np.array(ics)
        ics[np.isnan(ics)] = 0
        # auto_corr = np.array(auto_corr)
        # auto_corr[np.isnan(auto_corr)] = 0

        stats.ICs = ics
        stats.mean_IC = np.mean(ics)
        # stats.auto_corr = np.mean(auto_corr)

        if len(ics) > 1:
            stats.IC_IR = np.mean(ics) / np.std(ics)
        stats.positive_IC_rate = np.sum(ics > 0) / len(ics)
        return stats

    @staticmethod
    def cal_bin_ret(signal, ret, top=None, cell=20, zdt: bool = True, zdt_top: np.array = None,
                    standardize: bool = False, demean: bool = True):
        signal[np.isnan(signal)] = 0
        if top is None:
            top = signal != 0
        z = [[] for _ in range(cell)]
        r = [[] for _ in range(cell)]
        if zdt:
            if zdt_top is None:
                zdt_top = np.zeros(top.shape)
                zdt_top[:] = True
        for i in range(len(signal)):
            sig = signal[i, top[i]].copy()
            rr = ret[i, top[i]].copy()
            se = (~np.isnan(sig)) & (~np.isnan(rr))
            if zdt:
                se = se & zdt_top[i, top[i]]
            if demean:
                rr[se] -= np.mean(rr[se])
            if standardize:
                sig[se] = sig[se] - np.mean(sig[se])
                sig[se] = sig[se] / np.std(sig[se])

            sig = sig[se]
            rr = rr[se]

            arg_sig = np.argsort(sig)
            pos = 0

            while pos < cell:
                if pos < cell - 1:
                    z[pos] += list(sig[arg_sig[int(len(sig) / cell * pos): int(len(sig) / cell * (pos + 1))]])
                    r[pos] += list(rr[arg_sig[int(len(sig) / cell * pos): int(len(sig) / cell * (pos + 1))]])
                else:
                    z[pos] += list(sig[arg_sig[int(len(sig) / cell * pos):]])
                    r[pos] += list(rr[arg_sig[int(len(sig) / cell * pos):]])
                pos += 1
        return z, r
