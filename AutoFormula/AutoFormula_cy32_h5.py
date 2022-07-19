# Copyright (c) 2022 Dai HBG

"""
h5 version
"""

import numpy as np
import sys
import datetime

from Tester.AutoTester import *
from Tester.SelectTester import *
from AutoFormula.FormulaTree_cy import *
from AutoFormula.SignalGenerator_cy32 import *


class AutoFormula_cy32_h5:
    def __init__(self, start_date: str, end_date: str, data: Data, height: int = 3, symmetric: bool = False):
        """
        :param start_date: 该公式树
        :param end_date:
        :param data:  Data实例
        :param height: 最大深度
        :param symmetric: 是否对称
        """
        self.height = height
        self.symmetric = symmetric
        self.start_date = start_date
        self.end_date = end_date
        self.tree_generator = FormulaTree()
        self.data = data
        # self.tree = self.tree_generator.init_tree(height=self.height, symmetric=self.symmetric, dim_structure='2_2')
        self.operation = SignalGenerator(data=data)
        self.formula_parser = FormulaParser()
        self.AT = AutoTester()
        self.ST = SelectTester()
        self.need_top = ['topn_2d', 'se_topn_2d', 'c_topn_2d', 'se_c_topn_2d', 'csmean_2d', 'csmean_3d',
                         'csrank_2d', 'csrank_3d', 'zscore_2d', 'zscore_3d']

    def cal_formula(self, tree: FormulaTree, start: int, end: int,
                    return_type: str = 'signal', select_add_top: np.array = None,
                    meta_signal: np.array = None) -> np.array:  # 递归计算公式树的值
        """
        :param tree: 需要计算的公式树
        :param start: 计算数据开始的日期
        :param end: 计算数据的结束日期
        :param return_type: 返回值形式
        :param select_add_top: 额外设置的已选股票池，可以是固定规则初步筛选过的，必须确保是01矩阵，否则不保证计算结果的正确性
        :param meta_signal: 外部传入的信号矩阵，必须可以排序
        :return: 返回计算好的signal矩阵
        """
        if return_type == 'signal':
            if tree.variable_type == 'data':
                if type(tree.name) == int or type(tree.name) == float:
                    return tree.name  # 直接挂载在节点上，但是应该修改成需要数字的就直接返回数字
                return h5py.File(self.data.data_path, 'r')[tree.name][start: end + 1]
            elif tree.variable_type == 'intra_data':
                if tree.num_1 is not None:
                    return h5py.File(self.data.data_path, 'r')[tree.name][start: end + 1, tree.num_1, :]
                else:
                    return h5py.File(self.data.data_path, 'r')[tree.name][start: end + 1]
            else:
                if tree.operation_type == '1':
                    ipt = self.cal_formula(tree.left, start=start, end=end, return_type=return_type)
                    if len(ipt.shape) == 2:
                        if tree.name + '_2d' in self.need_top:
                            if meta_signal is None:
                                if select_add_top is None:
                                    return self.operation.operation_dic[tree.name + '_2d'](ipt, start)
                                return self.operation.operation_dic[tree.name + '_2d'](ipt *
                                                                                       select_add_top[start: end+1],
                                                                                       start)
                            return self.operation.operation_dic[tree.name + '_2d'](meta_signal[start: end+1], start)
                        else:
                            return self.operation.operation_dic[tree.name + '_2d'](ipt)
                    elif len(ipt.shape) == 3:
                        if tree.name + '_3d' in self.need_top:
                            return self.operation.operation_dic[tree.name + '_3d'](ipt, start)
                        else:
                            return self.operation.operation_dic[tree.name + '_3d'](ipt)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '1_num':
                    ipt = self.cal_formula(tree.left, start=start, end=end, return_type=return_type)
                    if len(ipt.shape) == 2:
                        if tree.name + '_2d' in self.need_top:
                            if meta_signal is None:
                                if select_add_top is None:
                                    return self.operation.operation_dic[tree.name + '_2d'](ipt, tree.num_1, start)
                                else:
                                    return self.operation.operation_dic[tree.name + '_2d'](
                                        ipt * select_add_top[start: end + 1], tree.num_1, start)
                            return self.operation.operation_dic[tree.name + '_2d'](meta_signal[start: end + 1],
                                                                                   tree.num_1, start)
                        else:
                            return self.operation.operation_dic[tree.name + '_2d'](ipt, tree.num_1)
                    elif len(ipt.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](ipt, tree.num_1)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '1_num_num':
                    ipt = self.cal_formula(tree.left, start=start, end=end, return_type=return_type)
                    if len(ipt.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](ipt, tree.num_1, tree.num_2)
                    elif len(ipt.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](ipt, tree.num_1, tree.num_2)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '1_num_num_num':
                    ipt = self.cal_formula(tree.left, start=start, end=end, return_type=return_type)
                    if len(ipt.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](ipt, tree.num_1, tree.num_2,
                                                                               tree.num_3)
                    elif len(ipt.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](ipt, tree.num_1, tree.num_2,
                                                                               tree.num_3)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '2':  # 此时需要判断有没有数字
                    if tree.num_1 is None:
                        input_1 = self.cal_formula(tree.left, start=start, end=end, return_type=return_type)
                        input_2 = self.cal_formula(tree.right, start=start, end=end, return_type=return_type)
                        if len(input_1.shape) == 2:
                            return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2)
                        elif len(input_1.shape) == 3:
                            return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2)
                    else:
                        # 暂时要求数字只能出现在后面
                        input_1 = self.cal_formula(tree.left, start=start, end=end, return_type=return_type)
                        if len(input_1.shape) == 2:
                            return self.operation.operation_dic[tree.name + '_num_2d'](input_1, tree.num_1)
                        elif len(input_1.shape) == 3:
                            return self.operation.operation_dic[tree.name + '_num_3d'](input_1, tree.num_1)
                        else:
                            raise NotImplementedError('input shape is not right!')
                        # if tree.left is not None:
                        #     return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic,
                        #                                                                     return_type),
                        #                                                    tree.num_1)
                        # else:
                        #     return self.operation.operation_dic[tree.name](tree.num_1,
                        #                                                    self.cal_formula(tree.right, data_dic,
                        #                                                                     return_type))
                if tree.operation_type == '2_num':
                    input_1 = self.cal_formula(tree.left, start=start, end=end, return_type=return_type)
                    input_2 = self.cal_formula(tree.right, start=start, end=end, return_type=return_type)
                    if len(input_1.shape) == 2:
                        if tree.name + '_2d' in self.need_top:  # 用在top类算子之前
                            if meta_signal is None:
                                if select_add_top is None:
                                    return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2, tree.num_1,
                                                                                           start)
                                return self.operation.operation_dic[tree.name + '_2d'](input_1 * select_add_top[start:
                                                                                                                end+1],
                                                                                       input_2, tree.num_1, start)
                            return self.operation.operation_dic[tree.name + '_2d'](meta_signal[start: end + 1],
                                                                                   input_2, tree.num_1, start)
                        else:
                            return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2, tree.num_1)
                    elif len(input_1.shape) == 3:
                        if tree.name + '_3d' in self.need_top:
                            return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2, tree.num_1, start)
                        else:
                            return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2, tree.num_1)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '2_num_num':
                    input_1 = self.cal_formula(tree.left, start=start, end=end, return_type=return_type)
                    input_2 = self.cal_formula(tree.right, start=start, end=end, return_type=return_type)
                    if len(input_1.shape) == 2:
                        if tree.name + '_2d' in self.need_top:
                            if meta_signal is None:
                                if select_add_top is None:
                                    return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2, tree.num_1,
                                                                                           tree.num_2, start)
                                return self.operation.operation_dic[tree.name + '_2d'](input_1 * select_add_top[start:
                                                                                                                end+1],
                                                                                       input_2,
                                                                                       tree.num_1, tree.num_2, start)
                            return self.operation.operation_dic[tree.name + '_2d'](meta_signal[start: end + 1],
                                                                                   input_2,
                                                                                   tree.num_1, tree.num_2, start)
                        return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2, tree.num_1, tree.num_2)
                    elif len(input_1.shape) == 3:
                        if tree.name + '_3d' in self.need_top:
                            return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2, tree.num_1,
                                                                                   tree.num_2, start)
                        return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2, tree.num_1, tree.num_2)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '2_num_num_num':
                    input_1 = self.cal_formula(tree.left, start=start, end=end, return_type=return_type)
                    input_2 = self.cal_formula(tree.right, start=start, end=end, return_type=return_type)
                    if len(input_1.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2, tree.num_1,
                                                                               tree.num_2, tree.num_3)
                    elif len(input_1.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2, tree.num_1,
                                                                               tree.num_2, tree.num_3)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '3':
                    input_1 = self.cal_formula(tree.left, start=start, end=end, return_type=return_type)
                    input_2 = self.cal_formula(tree.middle, start=start, end=end, return_type=return_type)
                    input_3 = self.cal_formula(tree.right, start=start, end=end, return_type=return_type)
                    if len(input_1.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2, input_3)
                    elif en(input_1.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2, input_3)
                    else:
                        raise NotImplementedError('input shape is not right!')

        if return_type == 'str':
            if tree.variable_type == 'data':
                return tree.name  # 返回字符串
            elif tree.variable_type == 'intra_data':  # 这里也需要判断是否有数字
                if tree.num_1 is not Nonr:
                    return '{' + tree.name + ',{}'.format(tree.num_1) + '}'
                else:
                    return '{' + tree.name + '}'
            else:
                if tree.operation_type == '1':
                    return tree.name + '{' + (self.cal_formula(tree.left, start=start, end=end,
                                                               return_type=return_type)) + '}'
                if tree.operation_type == '1_num':
                    return tree.name + '{' + self.cal_formula(tree.left, start=start, end=end,
                                                              return_type=return_type) + ',' + str(
                        tree.num_1) + '}'
                if tree.operation_type == '1_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, start=start, end=end,
                                                              return_type=return_type) + ',' + str(
                        tree.num_1) + ',' + str(tree.num_2) + '}'
                if tree.operation_type == '1_num_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, start=start, end=end,
                                                              return_type=return_type) + ',' + str(
                        tree.num_1) + ',' + str(tree.num_2) + ',' + str(tree.num_3) + '}'
                if tree.operation_type == '2':  # 此时需要判断是否有数字
                    if tree.num_1 is not None:
                        return tree.name + '{' + self.cal_formula(tree.left, start=start, end=end,
                                                                  return_type=return_type) + ',' + \
                               self.cal_formula(tree.right, start=start, end=end,
                                                return_type=return_type) + '}'
                    else:
                        if tree.left is not None:
                            return tree.name + '{' + self.cal_formula(tree.left, start=start, end=end,
                                                                      return_type=return_type) + ',' + \
                                   str(tree.num_1) + '}'
                        else:
                            return tree.name + '{' + str(tree.num_1) + ',' + \
                                   self.cal_formula(tree.right, start=start, end=end,
                                                    return_type=return_type) + '}'
                if tree.operation_type == '2_num':
                    return tree.name + '{' + self.cal_formula(tree.left, start=start, end=end,
                                                              return_type=return_type) + ',' + \
                           self.cal_formula(tree.right, start=start, end=end,
                                            return_type=return_type) + ',' + \
                           str(tree.num_1) + '}'
                if tree.operation_type == '2_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, start=start, end=end,
                                                              return_type=return_type) + ',' + \
                           self.cal_formula(tree.right, start=start, end=end,
                                            return_type=return_type) + ',' + \
                           str(tree.num_1) + ',' + str(tree.num_2) + '}'
                if tree.operation_type == '2_num_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, start=start, end=end,
                                                              return_type=return_type) + ',' + \
                           self.cal_formula(tree.right, start=start, end=end,
                                            return_type=return_type) + ',' + \
                           str(tree.num_1) + ',' + str(tree.num_2) + ',' + str(tree.num_3) + '}'
                if tree.operation_type == '3':
                    return tree.name + '{' + self.cal_formula(tree.left, start=start, end=end,
                                                              return_type=return_type) + ',' + \
                           self.cal_formula(tree.middle, start=start, end=end,
                                            return_type=return_type) + ',' + \
                           self.cal_formula(tree.right, start=start, end=end,
                                            return_type=return_type) + '}'

    def test_formula(self, formula: str, data: Data, start_date: str = None, end_date: str = None, shift: int = 1,
                     prediction_mode: str = False, fix_weekday: bool = None, zdt: bool = True,
                     select_univ_top: np.array = None, sec_type: str = 'stock'):
        """
        :param formula: 需要测试的因子表达式，如果是字符串形式，需要先解析成树
        :param start_date: 如果不提供则按照Data类默认的来
        :param end_date: 如果不提供则按照Data类默认的来
        :param prediction_mode: 是否是最新预测模式，是的话不需要测试，只生成signal
        :param fix_weekday: 指定统计哪些日期的信号
        :param shift: 平移天数
        :param zdt: 是否过滤涨跌停
        :param select_univ_top: 缩减univ，必须传入bool类型矩阵
        :param sec_type: 资产类型
        :return: 返回统计值以及该因子产生的信号矩阵
        """
        if not prediction_mode:
            if type(formula) == str:
                formula = self.formula_parser.parse(formula)

            if start_date is None:
                start_date = str(data.start_date)
            if end_date is None:
                end_date = str(data.end_date)

            start = self.data.get_real_date(start_date, direction='forward')
            end = self.data.get_real_date(end_date, direction='backward')

            af_start = start - 50 if start >= 50 else 0

            if type(formula) == str:
                formula = self.formula_parser.parse(formula)
            sig = self.cal_formula(formula, start=af_start, end=end)
            signal = np.zeros(self.data.ret.shape, dtype=np.float32)
            signal[af_start: end + 1] = sig

            if start > 50:  # 为了保证回溯算子有数据，暂时需要给start做移动
                start = start - 50
            else:
                start = 0

            if select_univ_top is None:
                if sec_type == 'stock':
                    return self.AT.test(signal[start:end + 1], self.data.ret[start + shift:end + shift+1],
                                        top=self.data.top[start:end + 1],
                                        zdt=zdt, zdt_top=self.data.zdt_top[start:end + 1]), signal
                if sec_type == 'conbond':
                    return self.AT.test(signal[start:end + 1], self.data.conbond_ret[start + shift:end + shift+1],
                                        top=self.data.conbond_top[start:end + 1],
                                        zdt=zdt, zdt_top=self.data.zdt_top[start:end + 1]), signal
            else:
                if sec_type == 'stock':
                    return self.AT.test(signal[start:end + 1], self.data.ret[start + shift:end + shift+1],
                                        top=self.data.top[start:end + 1] & select_univ_top[start:end + 1],
                                        zdt=zdt, zdt_top=self.data.zdt_top[start:end + 1]), signal
                if sec_type == 'conbond':
                    return self.AT.test(signal[start:end + 1], self.data.conbond_ret[start + shift:end + shift+1],
                                        top=self.data.conbond_top[start:end + 1] & select_univ_top[start:end + 1],
                                        zdt=zdt, zdt_top=self.data.zdt_top[start:end + 1]), signal
        else:
            if type(formula) == str:
                formula = self.formula_parser.parse(formula)
            sig = self.cal_formula(formula, start=0, end=len(data.ret))

            return sig

    def select_test_formula(self, formula: str, data: Data, zdt: bool = True,
                            start_date: str = None, end_date: str = None,
                            shift: int = 0, select_add_top: np.array = None,
                            meta_signal: np.array = None, sec_type: str = 'stock'):
        """
        :param formula: 需要测试的因子表达式，如果是字符串形式，需要先解析成树
        :param data: Data类
        :param start_date: 如果不提供则按照Data类默认的来
        :param end_date: 如果不提供则按照Data类默认的来
        :param zdt: 是否过滤涨跌停
        :param shift: 平移
        :param select_add_top: 额外设置的已选股票池，可以是固定规则初步筛选过的，必须确保是01矩阵，否则不保证计算结果的正确性
        :param meta_signal: 外部传入的信号矩阵，必须可以排序
        :param sec_type: 资产类型
        :return: 返回统计值以及该因子产生的信号矩阵
        """

        if start_date is None:
            start_date = str(data.start_date)
        if end_date is None:
            end_date = str(data.end_date)

        start = self.data.get_real_date(start_date, direction='forward')
        end = self.data.get_real_date(end_date, direction='backward')

        af_start = start - 50 if start >= 50 else 0

        if type(formula) == str:
            formula = self.formula_parser.parse(formula)
        sig = self.cal_formula(formula, start=af_start, end=end, select_add_top=select_add_top,
                               meta_signal=meta_signal)

        signal = np.zeros(data.ret.shape, dtype=np.float32)
        if select_add_top is None:
            signal[af_start: end + 1] = sig
        else:
            signal[af_start: end + 1] = sig * select_add_top[af_start: end + 1]

        ret = np.zeros(self.data.ret.shape, dtype=np.float32)
        if shift == 0:
            if sec_type == 'stock':
                ret[:] = self.data.ret[:]
            if sec_type == 'conbond':
                ret[:] = self.data.conbond_ret[:]
        else:
            if sec_type == 'stock':
                ret[:-shift] = self.data.ret[shift:]
            if sec_type == 'conbond':
                ret[:-shift] = self.data.conbond_ret[shift:]

        if zdt:
            zdt_top = np.zeros(self.data.top.shape)
            if shift == 0:
                zdt_top[1:] = self.data.zdt_top[:-1]
            elif shift == 1:
                zdt_top[:] = self.data.zdt_top[:]
            else:
                zdt_top[:-shift + 1] = self.data.zdt_top[shift - 1:]
            zdt_top = zdt_top > 0
        else:  # 暂时因为懒，无论如何都要过滤涨跌停
            zdt_top = np.zeros(data.top.shape)
            if shift == 0:
                zdt_top[1:] = self.data.zdt_top[:-1]
            elif shift == 1:
                zdt_top[:] = self.data.zdt_top[:]
            else:
                zdt_top[:-shift + 1] = self.data.zdt_top[shift - 1:]
            zdt_top = zdt_top > 0

        if sec_type == 'stock':
            return self.ST.test(signal=signal, ret=ret, top=self.data.top, zdt_top=zdt_top,
                                position_date_dic=self.data.position_date_dic,
                                order_code_dic=self.data.order_code_dic, s=start, e=end), signal

        if sec_type == 'conbond':
            return self.ST.test(signal=signal, ret=ret, top=self.data.conbond_top, zdt_top=zdt_top,
                                position_date_dic=self.data.position_date_dic,
                                order_code_dic=self.data.order_conbond_dic, s=start, e=end), signal
