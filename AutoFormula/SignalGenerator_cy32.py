# Copyright (c) 2022 Dai HBG

"""
cython version
"""

import h5py
import sys

sys.path.append('C:/Users/18316/Desktop/Repositories/Daily-Frequency-Quant/QBG')
sys.path.append('C:/users/18316/Desktop/cy_lib')

from AutoFormula.operations_cy32.one import *
# from AutoFormula.operations_cy32.one_num import *
# from AutoFormula.operations_cy32.one_num_num import *
# from AutoFormula.operations_cy32.one_num_num_num import *
from AutoFormula.operations_cy32.two import *
# from AutoFormula.operations_cy32.two_num import *
# from AutoFormula.operations_cy32.two_num_num import *
# from AutoFormula.operations_cy32.two_num_num_num import *
# from AutoFormula.operations_cy32.three import *

from DataLoader.Data import *


class SignalGenerator:
    def __init__(self, data: Data):
        """
        :param data: Data类的实例
        :param ind_name: 行业分类名字
        """
        self.operation_dic = {}
        self.get_operation()
        self.data = data

        # 单独注册需要用到额外信息的算子
        # self.operation_dic['zscore_2d'] = self.zscore_2d
        # self.operation_dic['csrank_2d'] = self.csrank_2d
        # self.operation_dic['csindneutral_2d'] = self.csindneutral_2d
        # self.operation_dic['csindneutral_3d'] = self.csindneutral_3d
        # self.operation_dic['csindmean_2d'] = self.csindmean_2d
        # self.operation_dic['csindsum_2d'] = self.csindsum_2d
        # self.operation_dic['csindmax_2d'] = self.csindmax_2d
        # self.operation_dic['csindmin_2d'] = self.csindmin_2d
        # self.operation_dic['csindstd_2d'] = self.csindstd_2d
        # self.operation_dic['csindmean_3d'] = self.csindmean_3d
        # self.operation_dic['csindsum_3d'] = self.csindsum_3d
        # self.operation_dic['csindmax_3d'] = self.csindmax_3d
        # self.operation_dic['csindmin_3d'] = self.csindmin_3d
        # self.operation_dic['csindstd_3d'] = self.csindstd_3d
        # self.operation_dic['truncate'] = self.truncate
        # self.operation_dic['marketbeta'] = self.marketbeta
        # self.operation_dic['discrete'] = self.discrete
        # self.operation_dic['topn_2d'] = self.topn_2d
        # self.operation_dic['se_topn_2d'] = self.se_topn_2d
        # self.operation_dic['c_topn_2d'] = self.c_topn_2d
        # self.operation_dic['se_c_topn_2d'] = self.se_c_topn_2d
        # self.operation_dic['csmean_2d'] = self.csmean_2d
        # self.operation_dic['csmean_3d'] = self.csmean_3d

    def get_operation(self):

        # 1型算符
        self.operation_dic['neg_2d'] = neg_2d
        self.operation_dic['neg_3d'] = neg_3d
        self.operation_dic['absv_2d'] = absv_2d
        self.operation_dic['absv_3d'] = absv_3d
        self.operation_dic['expv_2d'] = expv_2d
        self.operation_dic['expv_3d'] = expv_3d
        self.operation_dic['logv_2d'] = logv_2d
        self.operation_dic['logv_3d'] = logv_3d

        # 2型运算符
        self.operation_dic['add_2d'] = add_2d
        self.operation_dic['add_num_2d'] = add_num_2d
        self.operation_dic['add_3d'] = add_3d
        self.operation_dic['add_num_3d'] = add_num_3d
        self.operation_dic['minus_2d'] = minus_2d
        self.operation_dic['minus_num_2d'] = minus_num_2d
        self.operation_dic['minus_3d'] = minus_3d
        self.operation_dic['minus_num_3d'] = minus_num_3d
        self.operation_dic['prod_2d'] = prod_2d
        self.operation_dic['prod_num_2d'] = prod_num_2d
        self.operation_dic['prod_3d'] = prod_3d
        self.operation_dic['prod_num_3d'] = prod_num_3d
        self.operation_dic['div_2d'] = div_2d_32
        self.operation_dic['div_num_2d'] = div_num_2d
        self.operation_dic['div_3d'] = div_3d
        self.operation_dic['div_num_3d'] = div_num_3d
