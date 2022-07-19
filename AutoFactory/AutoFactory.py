# Copyright (c) 2022 Dai HBG

"""
Integrate data manager and factor testing.
"""

import numpy as np
import os
import pickle
import sys

sys.path.append('C:/Users/18316/Desktop/Repositories/Daily-Frequency-Quant/QBG')
from DataLoader.DataLoader import *
from DataLoader.DataLoader_h5 import *
from Tester.BackTester import *
from AutoFormula.AutoFormula import *
# from AutoFormula.AutoFormula_cy import *
# from AutoFormula.AutoFormula_cy32 import *
from AutoFormula.AutoFormula_cy32_h5 import *
# from AutoFormula.AutoFormula_cp32 import *
from Model.Model import *


class AutoFactory:
    def __init__(self, user_id, password, start_date, end_date, log_in: bool = False,
                 back_windows: int = 10, forward_windows: int = 10,
                 back_test_name='default', return_type='close-close-1', data_type: list = None,
                 data_path: str = None, back_test_data_path=None,
                 af_type: str = 'cython', data_mode: str = 'dic', univ: list = None, boards: list = None,
                 sec_type: str = 'stock'):
        """
        :param user_id: 登录聚宽的用户id
        :param password: 登录密码
        :param start_date: 总体的开始日期
        :param end_date: 总体的结束日期
        :param log_in: 是否登入DataLoader
        :param back_windows: 回溯日期长度
        :param back_test_name: 回测名称
        :param af_type: 算子类型，默认cython
        :param return_type: 收益率预测形式，默认是收盘价到收盘价，意味着日度调仓
        :param back_windows: 开始时间向前多长的滑动窗口
        :param forward_windows: 开始时间向后多长的滑动窗口
        :param data_mode: 数据模式，dic表示数据全部加载到内存中，h5表示使用h5格式存储
        :param univ: univ的选取
        :param boards: 构造top的板块
        :param sec_type: 交易的资产种类，可选stock和conbond
        """
        self.start_date = start_date
        self.end_date = end_date
        if data_path is None:
            data_path = 'D:/Documents/AutoFactoryData'
        self.data_path = data_path
        if back_test_data_path is None:
            back_test_data_path = 'D:/Documents/AutoFactoryData/BackTestData'
        self.back_test_data_path = back_test_data_path
        self.end_date = end_date
        if data_type is None:  # 需要获取的数据类型
            if sec_type == 'stock':
                data_type = ['stock_daily_pv', 'stock_daily_fundamental', 'stock_daily_money_flow',
                             'stock_daily_ext']
            elif sec_type == 'conbond':
                data_type = ['stock_daily_pv', 'conbond_daily_pv']
            else:
                raise NotImplementError

        self.data_mode = data_mode
        if data_mode == 'h5':
            self.dataloader = DataLoader_h5(user_id, password, data_path=self.data_path, log_in=log_in,
                                            back_test_data_path=self.back_test_data_path, data_type='float32')
        else:
            if af_type == 'cython':
                self.dataloader = DataLoader(user_id, password, data_path=self.data_path, log_in=log_in,
                                             back_test_data_path=self.back_test_data_path, data_type='float64')
            elif af_type == 'cython32':
                self.dataloader = DataLoader(user_id, password, data_path=self.data_path, log_in=log_in,
                                             back_test_data_path=self.back_test_data_path, data_type='float32')
            else:
                raise NotImplementedError
        if univ is None:
            univ = ['main_board']
        if boards is None:
            boards = univ.copy()
        for b in boards:  # top的构造必须包含在选择板块中
            assert (b in univ)
        self.data = self.dataloader.get_matrix_data(back_test_name=back_test_name, start_date=start_date,
                                                    end_date=end_date, back_windows=back_windows,
                                                    forward_windows=forward_windows,
                                                    return_type=return_type, data_type=data_type,
                                                    univ=univ, boards=boards)

        self.af_type = af_type
        if data_mode == 'h5':
            self.autoformula = AutoFormula_cy32_h5(start_date=start_date, end_date=self.end_date, data=self.data)
        else:
            if af_type == 'cython':
                self.autoformula = AutoFormula_cy(start_date=start_date, end_date=self.end_date, data=self.data)
            elif af_type == 'cython32':
                self.autoformula = AutoFormula_cy32(start_date=start_date, end_date=self.end_date, data=self.data)
            elif af_type == 'numpy':
                self.autoformula = AutoFormula(start_date=start_date, end_date=self.end_date, data=self.data)
            else:
                raise NotImplementedError('af_type should be cython or numpy!')

        self.autoformula_gpu = AutoFormula_gpu(start_date=start_date, end_date=self.end_date, data=self.data)

        # self.dsc = DataSetConstructor(self.data, signal_path=self.dump_signal_path)

    def set_ind(self, ind_name: str):  # 设定行业
        self.data.set_ind(ind_name=ind_name)
        self.autoformula.operation.data = self.data
        self.autoformula.data = self.data

    def reset_data(self):  # 只要涉及到重置data操作，就调用此方法
        # self.back_tester = BackTester(data=self.data)
        if self.af_type == 'cython':
            self.autoformula = AutoFormula_cy(start_date=self.start_date, end_date=self.end_date, data=self.data)
        elif self.af_type == 'numpy':
            self.autoformula = AutoFormula(start_date=self.start_date, end_date=self.end_date, data=self.data)
        # self.dsc = DataSetConstructor(self.data, signal_path=self.dump_signal_path)

    def reset_top(self, name):
        self.data.set_top(name)
        self.reset_data()

    def reset_ret(self, ret, data_type: str = 'float64'):
        self.data.set_ret(ret, data_type)
        self.reset_data()

    def test_factor(self, formula, start_date=None, end_date=None, prediction_mode=False, verbose: bool = False,
                    shift: int = 1, fix_weekday=None, zdt: bool = True, device: str = 'cpu', sec_type: str = 'stock',
                    select_add_top: np.array = None):  # 测试因子
        """
        :param formula: 回测的公式
        :param start_date: 回测开始日期
        :param end_date: 回测结束日期
        :param verbose: 是否打印
        :param prediction_mode: 是否是最新预测模式，是的话不需要测试，只生成signal
        :param fix_weekday: 指定统计哪些日期的信号
        :param shift: 平移天数
        :param zdt: 是否过滤zdt
        :param device: 使用cpu还是gpu进行计算
        :param sec_type: 资产类型
        :param select_add_top: 可选限制的top
        :return: 返回统计类的实例
        """
        if not prediction_mode:
            if start_date is None:
                start_date = self.start_date
            if end_date is None:
                end_date = self.end_date

            if self.data_mode == 'h5':
                if device == 'cpu':
                    stats, signal = self.autoformula.test_formula(formula, self.data, start_date, end_date, shift=shift,
                                                                  fix_weekday=fix_weekday, zdt=zdt, sec_type=sec_type)
                elif device == 'gpu':
                    stats, signal = self.autoformula_gpu.test_formula(formula, self.data, start_date, end_date,
                                                                      shift=shift,
                                                                      fix_weekday=fix_weekday, zdt=zdt,
                                                                      sec_type=sec_type,)
                else:
                    raise NotImplementedError
            else:
                stats, signal = self.autoformula.test_formula(formula, self.data, start_date, end_date, shift=shift,
                                                              fix_weekday=fix_weekday, zdt=zdt,)
            # return stats,s,e
            if verbose:
                print(start_date, end_date)
                print('mean IC: {:.4f}, positive_IC_rate: {:.4f}, IC_IR: {:.4f}'. \
                      format(stats.mean_IC, stats.positive_IC_rate, stats.IC_IR))
            return stats, signal
        else:
            return self.autoformula.test_formula(formula, self.data, start_date, end_date,
                                                 prediction_mode=prediction_mode)  # 只返回signal

    def select_test_factor(self, formula, start_date=None, end_date=None, zdt: bool = True, shift: int = 0,
                           verbose: bool = True, select_add_top: np.array = None,
                           meta_signal: np.array = None, device: str = 'cpu', sec_type: str = 'stock'):  # 测试因子
        """
        :param shift:
        :param formula: 回测的公式
        :param start_date: 回测开始日期
        :param end_date: 回测结束日期
        :param zdt: 是否过滤zdt
        :param verbose: 是否打印
        :param select_add_top: 额外设置的已选股票池，可以是固定规则初步筛选过的，必须确保是01矩阵，否则不保证计算结果的正确性
        :param meta_signal: 可以选择传入一个信号矩阵，例如ML的预测
        :param device: 使用cpu或者gpu计算
        :param sec_type: 资产类型
        :return: 返回统计类的实例
        """
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        if verbose:
            print(start_date, end_date)
        if device == 'cpu':
            stats, signal = self.autoformula.select_test_formula(formula, self.data, zdt=zdt, start_date=start_date,
                                                                 end_date=end_date,
                                                                 shift=shift, select_add_top=select_add_top,
                                                                 meta_signal=meta_signal, sec_type=sec_type)
        elif device == 'gpu':
            stats, signal = self.autoformula_gpu.select_test_formula(formula, self.data, zdt=zdt, start_date=start_date,
                                                                     end_date=end_date,
                                                                     shift=shift, select_add_top=select_add_top,
                                                                     meta_signal=meta_signal, sec_type=sec_type)
        else:
            raise NotImplementedError
        # return stats,s,e
        if verbose:
            print('mean ret: {:.4f}, sharpe_ratio: {:.4f}, win_rate: {:.4f}'. \
                  format(stats.mean_ret, stats.sharpe_ratio, stats.win_rate))
        return stats, signal
