# Copyright (c) 2022 Dai HBG


"""
该脚本用于读取本地的股票纯日频数据并返回data_dic

日志
2022-06-11
-更新：由于univ不一定是全股票，因此需要判断股票是否在univ中
2022-06-07
- 更新：加载涨跌停价格
2022-02-18
- 更新：读取文件改为dic，以提高速度
2022-02-16
- 更新：新增float32模式
2022-01-20
- 修复错误：数值空值要填充nan，并且停牌股要手动填充nan
"""


import pandas as pd
import numpy as np
import pickle


def get_stock_daily_pv(dates: np.array, date_position_dic: dict, code_order_dic: dict, data_path: str,
                       data_type: str = 'float64') -> dict:
    """
    :param dates: 所有日期的array
    :param date_position_dic: 日期到位置的字典
    :param code_order_dic: 股票代码到位置的字典
    :param data_path: 数据路径
    :param data_type: 数据格式
    """
    print('getting stock daily pv...')
    names = ['open', 'close', 'high', 'low', 'avg', 'factor', 'volume', 'high_limit', 'low_limit']

    if data_type == 'float64':
        data_dic = {name: np.zeros((len(dates), np.max(list(code_order_dic.values()))+1)) for name in names}  # 原始数据字典
    elif data_type == 'float32':
        data_dic = {name: np.zeros((len(dates), np.max(list(code_order_dic.values()))+1),
                                   dtype=np.float32) for name in names}  # 原始数据字典
    else:
        raise NotImplementedError
    for name in names:
        data_dic[name][:] = np.nan  # 先全部置为nan
    univ = set(code_order_dic.keys())  # 所有股票univ
    for date in dates:
        # data = pd.read_csv('{}/StockDailyData/{}/stock.csv'.format(data_path, date))
        with open('{}/StockDailyData/{}/stock.pkl'.format(data_path, date), 'rb') as f:
            data = pickle.load(f)
        index = data['code']
        k = date_position_dic[date]
        for j in range(len(index)):
            if index[j] not in univ:
                continue
            for name in names:
                if data['paused'][j] == 1:  # 略过停牌
                    continue
                data_dic[name][k, code_order_dic[index[j]]] = data[name][j]
        print('{} done.'.format(date))
    return data_dic
