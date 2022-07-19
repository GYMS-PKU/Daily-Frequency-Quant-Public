# Copyright (c) 2022 Dai HBG


"""
该脚本用于读取本地的股票日内量价数据并返回data_dic

日志
2022-06-11
-更新：由于univ不一定是全股票，因此需要判断股票是否在univ中
2022-02-18
- 更新：使用读取字典的方式进行数据读取
2022-02-16
- 新增float32格式支持
2022-01-20
- 日内数据获取时需要使用float32类型
"""
import os

import pandas as pd
import numpy as np
import pickle


def get_stock_intra_pv(dates: np.array, date_position_dic: dict, code_order_dic: dict, frequency: int = 1,
                       data_path: str = None, data_type: str = 'float64') -> dict:
    """
    :param dates: 所以日期的array
    :param date_position_dic: 日期到位置的字典
    :param code_order_dic: 股票代码到位置的字典
    :param frequency: 频率，表示几分钟数据
    :param data_path: 数据路径
    :param data_type: 数据格式
    """
    print('getting {}m data...'.format(frequency))
    names = ['intra_open', 'intra_high', 'intra_low', 'intra_close', 'intra_volume', 'intra_money',
             'intra_avg']  # 为了减少内存占用暂时只用这几个
    if frequency > 240:
        print('frequency should not be bigger than 240.')
    if data_type == 'float64':
        data_dic = {name: np.zeros((len(dates), 240 // frequency, len(code_order_dic))) for name in names}  # 原始数据字典
    elif data_type == 'float32':
        data_dic = {name: np.zeros((len(dates), 240 // frequency, len(code_order_dic)), dtype=np.float32)
                    for name in names}  # 原始数据字典
    else:
        raise NotImplementedError
    start_pos = [i for i in range(0, 240, frequency)]
    end_pos = [i for i in range(frequency - 1, 240, frequency)]
    for date in dates:
        with open('{}/StockIntraDayData/1m/{}/intra_data.pkl'.format(data_path, date), 'rb') as f:
            intra_data = pickle.load(f)
        stocks = list(set(code_order_dic.keys()) & set(intra_data.keys()))
        if data_type == 'float64':
            for stock in stocks:  # 依次读入高频数据
                data_dic['intra_open'][date_position_dic[date], :, code_order_dic[stock]] = \
                    intra_data[stock]['open'][start_pos].astype(np.float64)
                data_dic['intra_close'][date_position_dic[date], :, code_order_dic[stock]] = \
                    intra_data[stock]['close'][end_pos].astype(np.float64)

                high = intra_data[stock]['high'][:(240 // frequency * frequency)]\
                    .reshape(-1, frequency).astype(np.float64)
                data_dic['intra_high'][date_position_dic[date], :, code_order_dic[stock]] = np.max(high, axis=1)

                low = intra_data[stock]['low'][:(240 // frequency * frequency)]\
                    .reshape(-1, frequency).astype(np.float64)
                data_dic['intra_low'][date_position_dic[date], :, code_order_dic[stock]] = np.min(low, axis=1)

                volume = intra_data[stock]['volume'][:(240 // frequency * frequency)]\
                    .reshape(-1, frequency).astype(np.float64)
                data_dic['intra_volume'][date_position_dic[date], :, code_order_dic[stock]] = np.sum(volume, axis=1)
        elif data_type == 'float32':
            for stock in stocks:  # 依次读入高频数据
                data_dic['intra_open'][date_position_dic[date], :, code_order_dic[stock]] = \
                    intra_data[stock]['open'][start_pos].astype(np.float32)
                data_dic['intra_close'][date_position_dic[date], :, code_order_dic[stock]] = \
                    intra_data[stock]['close'][end_pos].astype(np.float32)

                high = intra_data[stock]['high'][:(240 // frequency * frequency)] \
                    .reshape(-1, frequency).astype(np.float32)
                data_dic['intra_high'][date_position_dic[date], :, code_order_dic[stock]] = np.max(high, axis=1)

                low = intra_data[stock]['low'][:(240 // frequency * frequency)] \
                    .reshape(-1, frequency).astype(np.float32)
                data_dic['intra_low'][date_position_dic[date], :, code_order_dic[stock]] = np.min(low, axis=1)

                volume = intra_data[stock]['volume'][:(240 // frequency * frequency)] \
                    .reshape(-1, frequency).astype(np.float32)
                data_dic['intra_volume'][date_position_dic[date], :, code_order_dic[stock]] = np.sum(volume, axis=1)

                money = intra_data[stock]['money'][:(240 // frequency * frequency)].reshape(-1, frequency)
                data_dic['intra_money'][date_position_dic[date], :, code_order_dic[stock]] = np.sum(money, axis=1)

                data_dic['intra_avg'][date_position_dic[date], :, code_order_dic[stock]] = \
                    data_dic['intra_money'][date_position_dic[date], :, code_order_dic[stock]] / \
                    data_dic['intra_volume'][date_position_dic[date], :, code_order_dic[stock]]
        else:
            raise NotImplementedError

        print('{} done.'.format(date))
    return data_dic
