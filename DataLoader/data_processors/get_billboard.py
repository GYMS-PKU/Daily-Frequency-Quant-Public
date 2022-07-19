# Copyright (c) 2022 Dai HBG

"""
读取龙虎榜数据并返回data_dic

日志
2022-04-06
- 初始化：初步的数据组织形式，生成5个（排名数量）买卖量（比率）的矩阵，正负号表示买入和卖出，缺失表示当日没有进入龙虎榜；生成一个类型矩阵；
"""

from jqdatasdk import *
import os
import numpy as np
import pandas as pd
import pickle


def get_billboard(dates: np.array, date_position_dic: dict, code_order_dic: dict, data_path: str,
                  data_type: str = 'float32') -> dict:
    """
    :param dates: 所有日期的array
    :param date_position_dic: 日期到位置的字典
    :param code_order_dic: 股票代码到位置的字典
    :param data_path: 数据路径
    :param data_type: 数据格式
    """
    print('getting billboard...')
    names = ['buy_1_v', 'buy_2_v', 'buy_3_v', 'buy_4_v', 'buy_5_v',
             'buy_1_r', 'buy_2_r', 'buy_3_r', 'buy_4_r', 'buy_5_r',
             'sell_1_v', 'sell_2_v', 'sell_3_v', 'sell_4_v', 'sell_5_v',
             'sell_1_r', 'sell_2_r', 'sell_3_r', 'sell_4_r', 'sell_5_r',
             'abnormal_code']  # 暂时不载入主体名称
    # 'sales_depart_name_b1', 'sales_depart_name_b2', 'sales_depart_name_b3', 'sales_depart_name_b4',
    # 'sales_depart_name_b5', 'sales_depart_name_s1', 'sales_depart_name_s2', 'sales_depart_name_s3',
    # 'sales_depart_name_s4', 'sales_depart_name_s5']

    if data_type == 'float64':
        data_dic = {name: np.zeros((len(dates), len(code_order_dic))) for name in names}  # 原始数据字典
    elif data_type == 'float32':
        data_dic = {name: np.zeros((len(dates), len(code_order_dic)), dtype=np.float32) for name in names}  # 原始数据字典
    else:
        raise NotImplementedError
    for name in names:
        data_dic[name][:] = np.nan  # 先全部置为nan
    for date in dates:
        with open('{}/StockDailyData/{}/billboard.pkl'.format(data_path, date), 'rb') as f:
            data = pickle.load(f)
        k = date_position_dic[date]
        for name in names:
            for key, value in data[name].items():
                try:
                    data_dic[name][k, code_order_dic[key]] = value
                except KeyError:
                    continue

        print('{} done.'.format(date))
    return data_dic
