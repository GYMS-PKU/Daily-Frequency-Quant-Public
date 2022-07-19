# Copyright (c) 2022 Dai HBG


"""
该脚本用于读取本地的期货纯日频数据并返回data_dic

日志
2022-06-04
- init
2022-06-05
- 为了和股票数据区分，期货数据的字段前必须加上fut
"""

import pandas as pd
import numpy as np
import pickle


def get_future_daily_pv(dates: np.array, date_position_dic: dict, future_order_dic: dict, data_path: str,
                        data_type: str = 'float32') -> dict:
    """
    :param dates: 所有日期的array
    :param date_position_dic: 日期到位置的字典
    :param future_order_dic: 期货代码到位置的字典，这里只有期货品种，没有月份合约
    :param data_path: 数据路径
    :param data_type: 数据格式
    """
    print('getting future daily pv...')
    names = ['open', 'close', 'high', 'low', 'avg', 'volume', 'money', 'pre_close', 'open_interest']

    if data_type == 'float64':
        data_dic = {f'fut_{name}': np.zeros((len(dates), len(future_order_dic),
                                             13)) for name in names}  # 期货最后一个维度放月份
    elif data_type == 'float32':  # 最后一个维度0是主力合约，1到12为对应月份的合约，如果没有，就全部都是nan
        data_dic = {f'fut_{name}': np.zeros((len(dates), len(future_order_dic), 13),
                                            dtype=np.float32) for name in names}
    else:
        raise NotImplementedError
    for name in names:
        data_dic[f'fut_{name}'][:] = np.nan  # 先全部置为nan
    for date in dates:
        with open('{}/FutureDailyData/{}/future.pkl'.format(data_path, date), 'rb') as f:
            data = pickle.load(f)
        index = data['code']
        k = date_position_dic[date]
        for j in range(len(index)):
            fut = index[j]
            tp = fut.split('.')[0][:-4]
            m = int(fut.split('.')[0][-2:])  # 月份
            for name in names:  # 期货没有停牌
                if m == 99:  # 主力合约
                    data_dic[f'fut_{name}'][k, future_order_dic[tp], 0] = data[name][j]
                elif m == 88:
                    continue
                else:
                    data_dic[f'fut_{name}'][k, future_order_dic[tp], m] = data[name][j]
        print('{} done.'.format(date))
    return data_dic
