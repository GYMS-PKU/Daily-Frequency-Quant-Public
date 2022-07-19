# Copyright (c) 2022 Dai HBG


"""
该脚本用于读取本地的股票纯日频数基本面据并返回data_dic

日志
2022-06-11
-更新：由于univ不一定是全股票，因此需要判断股票是否在univ中
2022-02-16
- 新增对float32的支持
"""

import pandas as pd
import numpy as np
import pickle


def get_stock_daily_fundamental(dates: np.array, date_position_dic: dict, code_order_dic: dict,
                                data_path: str, data_type: str = 'float64') -> dict:
    """
    :param data_path:
    :param dates: 所以日期的array
    :param date_position_dic: 日期到位置的字典
    :param code_order_dic: 股票代码到位置的字典
    :param data_type: 数据格式
    """
    univ = set(code_order_dic.keys())  # 所有股票univ
    print('getting stock daily fundamental...')
    # names = ['pe_ratio',  # 市盈率，每股市价为每股收益的倍数
    #          'turnover_ratio',  # 换手率
    #          'pb_ratio',  # 市净率，每股股价与每股净资产的比率
    #          'ps_ratio',  # 市销率，股票价格与每股销售收入之比
    #          'pcf_ratio',  # 市现率，每股市价为每股现金净流量
    #          'capitalization',  # 总股本
    #          'market_cap',  # 总市值
    #          'circulating_cap',  # 流通股本
    #          'circulating_market_cap',  # 流通市值
    #          'pe_ratio_lyr',
    #          'eps',  # 每股收益
    #          'adjusted_profit',  # 扣非利润
    #          'operating_profit', 'value_change_profit', 'roe', 'inc_return',
    #          'roa', 'net_profit_margin', 'gross_profit_margin', 'expense_to_total_revenue',
    #          'operation_profit_to_total_revenue', 'net_profit_to_total_revenue',
    #          'operating_expense_to_total_revenue', 'ga_expense_to_total_revenue',
    #          'financing_expense_to_total_revenue', 'operating_profit_to_profit',
    #          'invesment_profit_to_profit', 'adjusted_profit_to_profit',
    #          'goods_sale_and_service_to_revenue', 'ocf_to_revenue',
    #          'ocf_to_operating_profit', 'inc_total_revenue_year_on_year', 'inc_total_revenue_annual',
    #          'inc_revenue_year_on_year', 'inc_revenue_annual', 'inc_operation_profit_year_on_year',
    #          'inc_operation_profit_annual', 'inc_net_profit_year_on_year', 'inc_net_profit_annual',
    #          'inc_net_profit_to_shareholders_year_on_year', 'inc_net_profit_to_shareholders_annual',
    #          'net_profit',  # 净利润
    #          'minority_profit', 'basic_eps', 'diluted_eps'
    #          ]
    names = ['turnover_ratio']  # 目前只要换手率

    if data_type == 'float64':
        data_dic = {name: np.zeros((len(dates), len(code_order_dic))) for name in names}  # 原始数据字典
    elif data_type == 'float32':
        data_dic = {name: np.zeros((len(dates), len(code_order_dic)), dtype=np.float32) for name in names}  # 原始数据字典
    else:
        raise NotImplementedError
    for date in dates:
        with open('{}/StockDailyData/{}/fundamental.pkl'.format(data_path, date), 'rb') as f:
            data = pickle.load(f)
        index = data['code']
        k = date_position_dic[date]
        for j in range(len(index)):
            if index[j] not in univ:
                continue
            for name in names:
                data_dic[name][k, code_order_dic[index[j]]] = data[name][j]
        print('{} done.'.format(date))
    return data_dic
