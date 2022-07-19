# Copyright (c) 2022 Dai HBG


"""
该脚本用于读取本地的股票日频资金流数据并返回data_dic

日志
2022-06-11
-更新：由于univ不一定是全股票，因此需要判断股票是否在univ中
2022-02-16
- 新增对float32格式的支持
2022-02-18
- 更新：数据读取改用pkl
"""

import pandas as pd
import numpy as np
import pickle


def get_stock_daily_money_flow(dates: np.array, date_position_dic: dict, code_order_dic: dict, data_path: str,
                               data_type: str = 'float64') -> dict:
    """
    :param data_path:
    :param dates: 所以日期的array
    :param date_position_dic: 日期到位置的字典
    :param code_order_dic: 股票代码到位置的字典
    :param data_type: 数据格式
    """
    print('getting stock daily money flow...')
    names = ['net_pct_main', 'net_pct_xl', 'net_pct_l', 'net_pct_m', 'net_pct_s',
             'net_amount_main', 'net_amount_xl', 'net_amount_l', 'net_amount_m', 'net_amount_s']

    if data_type == 'float64':
        data_dic = {name: np.zeros((len(dates), len(code_order_dic))) for name in names}  # 原始数据字典
    elif data_type == 'float32':
        data_dic = {name: np.zeros((len(dates), len(code_order_dic)), dtype=np.float32) for name in names}  # 原始数据字典
    else:
        raise NotImplementedError
    univ = set(code_order_dic.keys())  # 所有股票univ
    for date in dates:
        with open('{}/StockDailyData/{}/money_flow.pkl'.format(data_path, date), 'rb') as f:
            data = pickle.load(f)
        index = data['sec_code']
        k = date_position_dic[date]
        for j in range(len(index)):
            if index[j] not in univ:
                continue
            for name in names:
                data_dic[name][k, code_order_dic[index[j]]] = data[name][j]
        print('{} done.'.format(date))
    return data_dic
