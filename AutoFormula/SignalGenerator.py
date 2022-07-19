# Copyright (c) 2021 Dai HBG

"""
numpy version
"""


import sys
sys.path.append('C:/Users/Administrator/Desktop/Daily-Frequency-Quant/QBG')
sys.path.append('C:/Users/HBG/Desktop/Daily-Frequency-Quant/QBG')

from AutoFormula.operations.one import *
# from AutoFormula.operations.one_num import *
# from AutoFormula.operations.one_num_num import *
# from AutoFormula.operations.one_num_num_num import *
from AutoFormula.operations.two import *
# from AutoFormula.operations.two_num import *
# from AutoFormula.operations.two_num_num import *
# from AutoFormula.operations.two_num_num_num import *

from DataLoader.Data import *


class SignalGenerator:
    def __init__(self, data: Data):
        """
        :param data: Data类的实例
        """
        self.operation_dic = {}
        self.get_operation()
        self.data = data

    def get_operation(self):

        # 1型算符
        self.operation_dic['neg'] = neg
        self.operation_dic['absv'] = absv
        self.operation_dic['intratsfftreal'] = intratsfftreal
        self.operation_dic['intratsfftimag'] = intratsfftimag

        # 2型运算符
        self.operation_dic['add'] = add
        self.operation_dic['minus'] = minus
        self.operation_dic['prod'] = prod
        self.operation_dic['div'] = div
        self.operation_dic['intratsregres'] = intratsregres
        self.operation_dic['lt'] = lt
        self.operation_dic['le'] = le
        self.operation_dic['gt'] = gt
        self.operation_dic['ge'] = ge
