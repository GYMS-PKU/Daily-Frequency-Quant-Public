# Copyright (c) 2022 Dai HBG


"""
该脚本用于读取本地的行业纯日频数据并返回industry_dic

日志
2022-02-16
- 新增对float32的支持
"""

import pickle
import numpy as np


def get_industry_daily(dates: np.array, date_position_dic: dict, code_order_dic: dict,
                       industry_order_dic: dict, data_path: str = None, data_type: str = 'float64') -> dict:
    """
    :param dates: 所以日期的array
    :param date_position_dic: 日期到位置的字典
    :param code_order_dic: 股票代码到位置的字典
    :param industry_order_dic: 行业代码到标号的字典
    :param data_path:
    :param data_type: 数据格式
    """
    ind_names = ['swf', 'sws', 'swt', 'jqf', 'jqs', 'zjw', 'concept']  # 行业分类准则
    # 初始化为-1，如果有股票不被分类
    industry_dic = {name: -np.ones((len(dates), len(code_order_dic))).astype(np.int32) for name in ind_names}

    for date in dates:
        with open('{}/StockDailyData/{}/industry.pkl'.format(data_path, date), 'rb') as f:
            data = pickle.load(f)
        k = date_position_dic[date]
        for name in ind_names:
            for ind_code, stocks in data[name].items():
                for stock in stocks:
                    try:
                        industry_dic[name][k, code_order_dic[stock]] = industry_order_dic[name][ind_code]
                    except KeyError:
                        pass
    return industry_dic
