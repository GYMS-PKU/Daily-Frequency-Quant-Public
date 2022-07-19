# Copyright (c) 2022 Dai HBG


"""
该脚本用于读取本地的可转债纯日频数据并返回data_dic

日志
2022-06-27
- init
"""

import pandas as pd
import numpy as np
import pickle


def get_conbond_daily_pv(dates: np.array, date_position_dic: dict, conbond_order_dic: dict, data_path: str,
                         data_type: str = 'float32') -> dict:
    """
    :param dates: 所有日期的array
    :param date_position_dic: 日期到位置的字典
    :param conbond_order_dic: 股票代码到位置的字典
    :param data_path: 数据路径
    :param data_type: 数据格式
    """
    print('getting conbond daily pv...')
    names = ['pre_close', 'open', 'close', 'high', 'low', 'volume', 'money']

    if data_type == 'float64':
        data_dic = {f'con_{name}': np.zeros((len(dates), np.max(list(conbond_order_dic.values()))+1)) for name in names}
    elif data_type == 'float32':
        data_dic = {f'con_{name}': np.zeros((len(dates), np.max(list(conbond_order_dic.values()))+1),
                                            dtype=np.float32) for name in names}
    else:
        raise NotImplementedError
    for name in names:
        data_dic[f'con_{name}'][:] = np.nan  # 先全部置为nan
    univ = set(conbond_order_dic.keys())  # 所有可转债univ
    for date in dates:
        # data = pd.read_csv('{}/StockDailyData/{}/stock.csv'.format(data_path, date))
        with open('{}/ConbondDailyData/{}/conbond.pkl'.format(data_path, date), 'rb') as f:
            data = pickle.load(f)
        index = data['code']
        k = date_position_dic[date]
        for j in range(len(index)):
            if index[j] not in univ:
                continue
            for name in names:
                data_dic[f'con_{name}'][k, conbond_order_dic[index[j]]] = data[name][j]
        print('{} done.'.format(date))
    return data_dic
