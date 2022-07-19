# Copyright (c) 2022 Dai HBG


"""
该脚本用于读取本地的股票纯日频拓展数据并返回data_dic，拓展数据包括
- 融资融券数据
- 集合竞价数据
- 北向资金数据
- 龙虎榜数据

日志
2022-06-11
-更新：由于univ不一定是全股票，因此需要判断股票是否在univ中
2022-02-16
- 新增对float32的支持
2022-02-18
- 更新：竞价数据改为读取字典
"""

import pandas as pd
import numpy as np
import pickle


def get_stock_daily_ext(dates: np.array, date_position_dic: dict, code_order_dic: dict, data_path: str,
                        data_type: str = 'float64') -> dict:
    """
    :param dates: 需要获取的数据
    :param date_position_dic:
    :param code_order_dic:
    :param data_path:
    :param data_type: 数据格式
    :return:
    """
    print('getting stock daily mtss...')
    # 获取融资融券数据
    names = ['fin_value', 'fin_buy_value', 'fin_refund_value', 'sec_value', 'sec_sell_value',
             'sec_refund_value', 'fin_sec_value']
    univ = set(code_order_dic.keys())  # 所有股票univ
    if data_type == 'float64':
        data_dic = {name: np.zeros((len(dates), len(code_order_dic))) for name in names}  # 原始数据字典
    elif data_type == 'float32':
        data_dic = {name: np.zeros((len(dates), len(code_order_dic)), dtype=np.float32) for name in names}  # 原始数据字典
    else:
        raise NotImplementedError
    for date in dates:
        with open('{}/StockDailyData/{}/mtss.pkl'.format(data_path, date), 'rb') as f:
            data = pickle.load(f)
        index = data['sec_code']
        k = date_position_dic[date]
        for j in range(len(index)):
            if index[j] not in univ:
                continue
            for name in names:
                data_dic[name][k, code_order_dic[index[j]]] = data[name][j]
        print('{} done.'.format(date))

    # 集合竞价
    print('getting stock daily auction...')
    names = ['current', 'volume', 'money', 'a1_p', 'a1_v', 'a2_p', 'a2_v', 'a3_p', 'a3_v', 'a4_p', 'a4_v',
             'a5_p', 'a5_v', 'b1_p', 'b1_v', 'b2_p', 'b2_v', 'b3_p', 'b3_v', 'b4_p', 'b4_v', 'b5_p', 'b5_v']
    for name in names:
        if data_type == 'float64':
            data_dic['auc_'+name] = np.zeros((len(dates), len(code_order_dic)))
        elif data_type == 'float32':
            data_dic['auc_' + name] = np.zeros((len(dates), len(code_order_dic)), dtype=np.float32)
        else:
            raise NotImplementedError
    for date in dates:
        # data = pd.read_csv('{}/StockDailyData/{}/auction.csv'.format(data_path, date))
        with open('{}/StockDailyData/{}/auction.pkl'.format(data_path, date), 'rb') as f:
            data = pickle.load(f)
        index = data['code']
        k = date_position_dic[date]
        for j in range(len(index)):
            if index[j] not in univ:
                continue
            for name in names:
                data_dic['auc_'+name][k, code_order_dic[index[j]]] = data[name][j]
        print('{} done.'.format(date))

    # 北向资金数据
    
    return data_dic
