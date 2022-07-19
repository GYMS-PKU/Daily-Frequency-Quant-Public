# Copyright (c) 2021-2022 Dai HBG

"""
BackTester类根据信号和收益率矩阵进行回测
开发日志
2021-09-07
-- 更新：BackTester类统计pnl序列的平均日收益，最大回撤，标准差，夏普比，最长亏损时间
2021-09-11
-- 修复：回测时剔除涨停板
2021-09-21
-- 修复：回测是要根据上一期的持仓确定涨停板是否剔除，有可能涨停的股票是有持仓的
2022-05-01
- 重写结构
"""

import numpy as np
import datetime
import sys
sys.path.append('C:/Users/Administrator/Desktop/Daily-Frequency-Quant/QBG')
sys.path.append('C:/Users/HBG/Desktop/Daily-Frequency-Quant/QBG')
from Tester.tools.risk_measure import *


class BackTester:
    def __init__(self, signal: np.array, target: np.array, univ: np.array, zt_top: np.array, dt_top: np.array,
                 fee: float = 0.00075):
        """
        :param signal: 信号矩阵，暂时默认已经每个截面的univ做了zscore标准化
        :param target: 目标矩阵，已经对齐
        :param univ: 股票池矩阵
        :param zt_top: 涨停矩阵
        :param dt_top: 跌停矩阵
        :param fee: 手续费比例
        """
        self.signal = signal
        self.target = target
        self.univ = univ
        self.dt_top = dt_top
        self.zt_top = zt_top
        self.fee = fee

        self.pnl = []  # pnl序列
        self.cumulated_pnl = []  # 累计pnl
        self.market_pnl = []  # 如果纯多头，这里存市场的pnl，以比较超额收益
        self.market_cumulated_pnl = []

        self.max_dd = 0  # 统计pnl序列的最大回撤
        self.mean_pnl = 0  # 平均pnl
        self.std = 0  # 统计pnl序列的标准差
        self.sharp_ratio = 0  # 夏普比
        self.max_loss_time = 0  # 最长亏损时间

        self.tvr = 0  # 平均换手率
        self.c_tvr = []

        self.log = []  # 记录每一个具体的交易日给出的股票

    def cal_stats(self):
        self.std = np.std(self.pnl)
        self.mean_pnl = np.mean(self.pnl)
        self.sharp_ratio = self.mean_pnl / self.std if self.std > 0 else 0

        max_d, max_loss_day, start, end = risk_measure(self.pnl)
        self.max_dd = max_d
        self.max_loss_time = max_loss_day

    def long_short(self):  # 多空策略
        """
        :return: None
        """
        self.pnl = []
        self.cumulated_pnl = [0]
        self.market_pnl = []
        self.market_cumulated_pnl = []

        real_position = np.zeros(self.signal.shape)  # 通过过滤涨跌停获得实际仓位
        # ideal_position = np.zeros(self.signal.shape)
        real_position[0, self.univ[0]] = self.signal[0, self.univ[0]]
        real_position[0, self.univ[0]] -= np.mean(real_position[0, self.univ[0]])
        real_position[0, self.univ[0] & (real_position[0] > 0)] /= np.sum(
            real_position[0, self.univ[0] & (real_position[0] > 0)])
        real_position[0, self.univ[0] & (real_position[0] < 0)] /= np.sum(
            -real_position[0, self.univ[0] & (real_position[0] < 0)])

        ideal_position[0] = real_position[0]

        real_position[0, (real_position[0] > 0) & (~self.zt_top[0])] = 0
        real_position[0, (real_position[0] < 0) & (~self.dt_top[0])] = 0
        for i in range(1, len(real_position)):
            real_position[i, self.univ[i]] = self.signal[i, self.univ[i]]
            real_position[i, self.univ[i]] -= np.mean(real_position[i, self.univ[i]])
            real_position[i, self.univ[i] & (real_position[i] > 0)] /= np.sum(
                real_position[i, self.univ[i] & (real_position[i] > 0)])
            real_position[i, self.univ[i] & (real_position[i] < 0)] /= np.sum(
                -real_position[i, self.univ[i] & (real_position[i] < 0)])

            real_position[i, (~self.zt_top[i]) | (~self.dt_top[i])] = real_position[i - 1, (~self.zt_top[i]) | (
                ~self.dt_top[i])] * \
                                                                      (1 + self.target[
                                                                          i - 1, (~self.zt_top[i]) | (~self.dt_top[i])])

        tvr = []
        self.c_tvr = [0]
        for i in range(len(real_position)):
            self.pnl.append(np.sum(real_position[i] * self.target[i]))

            if i >= 1:
                self.pnl[-1] -= self.fee * np.sum(np.abs((real_position[i] - real_position[i - 1])))
                tvr.append(np.sum(np.abs((real_position[i] - real_position[i - 1]))))
                self.c_tvr.append(self.c_tvr[-1] + tvr[-1])
            self.cumulated_pnl.append(self.cumulated_pnl[-1] + self.pnl[-1])
        self.tvr = np.mean(tvr)

        self.cal_stats()

    def long(self):  # 多头
        """
        :return: None
        """
        self.pnl = []
        self.cumulated_pnl = [0]
        self.market_pnl = []
        self.market_cumulated_pnl = []

        real_position = np.zeros(self.signal.shape)  # 通过过滤涨跌停获得实际仓位
        ideal_position = np.zeros(self.signal.shape)
        real_position[0, self.univ[0]] = self.signal[0, self.univ[0]]
        real_position[0, self.univ[0]] -= np.mean(real_position[0, self.univ[0]])
        real_position[0, self.univ[0] & (real_position[0] > 0)] /= np.sum(
            real_position[0, self.univ[0] & (real_position[0] > 0)])
        real_position[0, self.univ[0] & (real_position[0] < 0)] = 0

        ideal_position[0] = real_position[0]

        real_position[0, (real_position[0] > 0) & (~self.zt_top[0])] = 0
        real_position[0, (real_position[0] < 0) & (~self.dt_top[0])] = 0

        for i in range(1, len(real_position)):
            real_position[i, self.univ[i]] = self.signal[i, self.univ[i]]
            real_position[i, self.univ[i]] -= np.mean(real_position[i, self.univ[i]])
            real_position[i, self.univ[i] & (real_position[i] > 0)] /= np.sum(
                real_position[i, self.univ[i] & (real_position[i] > 0)])
            real_position[i, self.univ[i] & (real_position[i] < 0)] = 0

            ideal_position[i] = real_position[i]

            real_position[i, (~self.zt_top[i]) | (~self.dt_top[i])] = real_position[i - 1, (~self.zt_top[i]) | (
                ~self.dt_top[i])] * \
                                                                      (1 + self.target[
                                                                          i - 1, (~self.zt_top[i]) | (~self.dt_top[i])])

        # self.real_position = real_position
        # self.ideal_position = ideal_position

        tvr = []
        self.c_tvr = [0]
        for i in range(len(real_position)):
            self.pnl.append(np.sum(real_position[i] * self.target[i]) - np.nanmean(self.target[i, self.univ[i]]))
            if i >= 1:
                self.pnl[-1] -= self.fee * np.sum(np.abs((real_position[i] - real_position[i - 1])))
                tvr.append(np.sum(np.abs((real_position[i] - real_position[i - 1]))))
                self.c_tvr.append(self.c_tvr[-1] + tvr[-1])
            self.cumulated_pnl.append(self.cumulated_pnl[-1] + self.pnl[-1])
        self.tvr = np.mean(tvr)

        self.cal_stats()

    def long_top_n(self, start_date=None, end_date=None, n=0, signal=None, zt_filter=True,
                   position_mode='weighted'):  # 做多预测得分最高的n只股票
        """
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param n: 做多多少只股票，默认按照top做多
        :param signal: 可以传入一个自定义的signal，默认使用自身的signal
        :param zt_filter: 是否过滤涨停
        :param position_mode: 仓位配置模式
        :return:
        """
        self.pnl = []
        self.cumulated_pnl = []
        self.market_pnl = []
        self.market_cumulated_pnl = []

        # 验证用功能：记录每一天具体的给出的股票代码和实际的收益率
        self.log = []

        if start_date is None:
            start_date = str(self.data.start_date)
        if end_date is None:
            end_date = str(self.data.end_date)

        start, end = self.data.get_real_date(start_date, end_date)
        if signal is None:
            signal = self.signal.copy()

        pos = np.array([i for i in range(len(self.data.top[0]))])  # 记录位置
        last_pos = None  # 记录上一次持仓的股票
        if n != 0:
            zt = []
            in_zt = []
            zt_ret = []
            zt_w = []
            for i in range(start, end + 1):
                tmp = signal[i].copy()
                tmp[self.data.top[i]] -= np.mean(tmp[self.data.top[i]])
                tmp[self.data.top[i] & (tmp > 0)] /= np.sum(tmp[self.data.top[i] & (tmp > 0)])
                tmp[self.data.top[i] & (tmp < 0)] = 0
                if np.sum(tmp != 0) == 0:
                    continue
                a = tmp[self.data.top[i] & (tmp > 0)].argsort()[-n:]
                this_pos = pos[self.data.top[i] & (tmp > 0)][a].copy()  # 本次持仓的股票
                '''
                try:
                    print(len(list(set(this_pos) & set(last_pos))))
                except TypeError:
                    pass
                '''
                self.log.append((self.data.position_date_dic[i],
                                 self.data.order_code_dic[pos[self.data.top[i] & (tmp > 0)][a][0]]))
                ret_tmp = self.data.ret[i + 1, self.data.top[i] & (tmp > 0)][a].copy()
                sig_tmp = tmp[self.data.top[i] & (tmp > 0)][a].copy()
                sig_tmp /= np.sum(sig_tmp)
                zt_weight = []
                if zt_filter:
                    in_z = 0
                    if last_pos is not None:
                        for j in range(n):
                            if (self.data.ret[i, self.data.top[i] & (tmp > 0)][a][j] > 0.099) and (this_pos[j]
                                                                                                   not in last_pos):
                                zt_weight.append(sig_tmp[j])
                                sig_tmp[j] = 0
                                this_pos[j] = -1
                            elif self.data.ret[i, self.data.top[i] & (tmp > 0)][j] > 0.099:
                                in_z += 1

                    else:
                        for j in range(n):
                            if self.data.ret[i, self.data.top[i] & (tmp > 0)][a][j] > 0.099:
                                sig_tmp[j] = 0
                                zt_weight.append(sig_tmp[j])
                                this_pos[j] = -1

                # sig_tmp /= np.sum(sig_tmp)
                # print(np.sum(sig_tmp == 0))
                    in_zt.append(in_z)
                zt.append(np.sum(sig_tmp == 0))  # 统计涨停总数
                if zt[-1] > 0:
                    if position_mode == 'weighted':
                        zt_ret.append(np.sum(np.array(zt_weight) * ret_tmp[sig_tmp == 0]))  # 统计涨停收益
                    else:
                        zt_ret.append(np.mean(ret_tmp[sig_tmp == 0]))  # * zt[-1] / n)
                    zt_w.append(np.sum(zt_weight))  # 涨停权重占比
                else:
                    zt_ret.append(0)
                    zt_w.append(0)
                last_pos = this_pos.copy()
                if position_mode == 'mean':
                    self.pnl.append(np.mean(ret_tmp[sig_tmp != 0]) -
                                    np.mean(self.data.ret[i + 1, self.data.top[i]]))
                else:
                    self.pnl.append(np.sum(sig_tmp * ret_tmp) -
                                    np.mean(self.data.ret[i + 1, self.data.top[i]]))  # 这是按照比例投资，只看纯超额
                if not self.cumulated_pnl:
                    """
                    self.cumulated_pnl.append(np.sum(tmp[self.data.top[i] & (tmp > 0)][a] *
                                                     self.data.ret[i + 1, self.data.top[i] & (tmp > 0)][a]) /
                                              np.sum(tmp[self.data.top[i] & (tmp > 0)][a]))
                                              """
                    # self.cumulated_pnl.append(np.mean(ret_tmp[ret_tmp != 0]))
                    self.cumulated_pnl.append(self.pnl[-1])  # 按照比例投资
                    self.market_cumulated_pnl.append(np.mean(self.data.ret[i + 1, self.data.top[i]]))
                else:
                    """
                    self.cumulated_pnl.append(np.sum(tmp[self.data.top[i] & (tmp > 0)][a] *
                                                     self.data.ret[i + 1, self.data.top[i] & (tmp > 0)][a]) /
                                              np.sum(tmp[self.data.top[i] & (tmp > 0)][a]))
                                              """
                    # self.cumulated_pnl.append(self.cumulated_pnl[-1] +
                    # np.mean(ret_tmp[ret_tmp != 0]))
                    self.cumulated_pnl.append(self.cumulated_pnl[-1] + self.pnl[-1])

                    self.market_cumulated_pnl.append(self.market_cumulated_pnl[-1] +
                                                     np.mean(self.data.ret[i + 1, self.data.top[i]]))
            print(np.mean(zt))
            zt_ret = np.array(zt_ret)
            zt = np.array(zt)
            print(np.mean(zt_ret) * 100)
            print(np.mean(zt_w))
            print(np.mean(in_zt))
            if position_mode == 'mean':
                print(np.corrcoef(zt_ret[zt != 0], zt[zt != 0])[0, 1])
            self.cal_stats()
            return zt_ret, zt
        else:
            for i in range(start, end + 1):
                tmp = self.signal[i].copy()
                tmp[self.data.top[i]] -= np.mean(tmp[self.data.top[i]])
                tmp[self.data.top[i] & (tmp > 0)] /= np.sum(tmp[self.data.top[i] & (tmp > 0)])
                tmp[self.data.top[i] & (tmp < 0)] = 0
                self.pnl.append(np.sum(tmp[self.data.top[i]] * self.data.ret[i + 1, self.data.top[i]]))
                self.market_pnl.append(np.mean(self.data.ret[i + 1, self.data.top[i]]))
                if not self.cumulated_pnl:
                    self.cumulated_pnl.append(np.sum(tmp[self.data.top[i]] * self.data.ret[i + 1,
                                                                                           self.data.top[i]]))
                    self.market_cumulated_pnl.append(np.mean(self.data.ret[i + 1, self.data.top[i]]))
                else:
                    self.cumulated_pnl.append(
                        self.cumulated_pnl[-1] + np.sum(tmp[self.data.top[i]] * self.data.ret[i + 1,
                                                                                              self.data.top[i]]))
                    self.market_cumulated_pnl.append(self.market_cumulated_pnl[-1] +
                                                     np.mean(self.data.ret[i + 1, self.data.top[i]]))
        self.cal_stats()
        return self.log

    def long_stock_predict(self, date=None, n=1, signal=None):  # 非回测模式，直接预测最新交易日的股票
        """
        :param date: 预测的日期，默认是最新的日期
        :param n: 需要预测多少只股票
        :param signal: 可以直接输入signal
        :return: 返回预测的股票代码以及他们的zscore分数
        """
        if signal is None:
            signal = self.signal.copy()
        pos = np.array([i for i in range(len(self.data.top[0]))])
        if date is None:
            start, end = self.data.get_real_date(str(self.data.start_date), str(self.data.end_date))
        else:
            start, end = self.data.get_real_date(date, date)
        for i in range(end, end + 1):
            tmp = signal[i].copy()
            tmp[self.data.top[i]] -= np.mean(tmp[self.data.top[i]])
            tmp[self.data.top[i] & (tmp > 0)] /= np.sum(tmp[self.data.top[i] & (tmp > 0)])
            tmp[self.data.top[i] & (tmp < 0)] = 0
            a = tmp[self.data.top[i] & (tmp > 0)].argsort()[-n:]
            return (self.data.position_date_dic[i],
                    [self.data.order_code_dic[pos[self.data.top[i] & (tmp > 0)][a][j]] for j in range(n)],
                    tmp[self.data.top[i] & (tmp > 0)][a])

    def generate_signal(self, model=None, signals_dic=None, start_date=None, end_date=None):
        """
        :param model: 一个模型
        :param signals_dic: 使用的原始信号字典
        :param start_date: 得到信号的开始日期
        :param end_date: 得到信号的结束日期
        :return:
        """
        # 支持传入模型预测得到signal测试，为了方便必须返回一个形状完全一致的signal矩阵，只不过可以只在对应位置有值
        if start_date is None:
            start_date = str(self.data.start_date)
        if end_date is None:
            end_date = str(self.data.end_date)
        signal = np.zeros(self.data.data_dic['close'].shape)

        start, end = self.data.get_real_date(start_date, end_date)

        if model is not None:
            for i in range(start, end + 1):
                tmp_x = []
                for j in signals_dic.keys():
                    tmp = signals_dic[j][i].copy()
                    tmp[np.isnan(tmp)] = 0
                    tmp[self.data.top[i]] -= np.mean(tmp[self.data.top[i]])
                    if np.sum(tmp[self.data.top[i]] != 0) >= 2:
                        tmp[self.data.top[i]] /= np.std(tmp[self.data.top[i]])
                    tmp_x.append(tmp)
                tmp_x = np.vstack(tmp_x).T  # 用于预测
                signal[i, self.data.top[i]] = model.predict(tmp_x[self.data.top[i], :])  # 只预测需要的部分
        else:
            signal = signals_dic[0].copy()
        self.signal = signal

        return signal
