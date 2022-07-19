# Copyright (c) 2022 Dai HBG


"""
构造top矩阵

日志
2022-06-28
- 更新：新增generate_conbond_top方法
2022-06-11
-更新：由于univ不一定是全股票，因此需要判断股票是否在univ中
2022-06-07
- 更新：股票和期货分为两个方法
- 更新：股票方法新增主板和创业板、科创版选择
2022-06-03
- 读取all_securities改用pickle，提高读取速度
2022-02-18
- 读取数据都改为读取pkl
2022-02-04
- 更新：剔除ST股的方法需要修改
2022-01-20
- 更新：生成普通top时需要剔除ST股
2022-01-11
- init
"""

import numpy as np
import pandas as pd
import pickle


# 生成某个指数的top
def generate_index_top(dates: np.array, date_position_dic: dict,
                       code_order_dic: dict, data_path: str, top_type: str = 'ZZ1000') -> np.array:
    """
    :param dates:
    :param date_position_dic:
    :param code_order_dic:
    :param data_path:
    :param top_type: top类型
    :return:
    """
    top = np.zeros((len(dates), len(code_order_dic)))
    for date in dates:
        with open('{}/StockDailyData/{}/index_dic.pkl'.format(data_path, date), 'rb') as f:
            index = pickle.load(f)
        for stock in index[top_type]:
            top[date_position_dic[date], code_order_dic[stock]] = 1
    top = top > 0
    return top


def generate_stock_top(dates: np.array, date_position_dic: dict,
                       code_order_dic: dict, data_path: str, top_type: str = 'listed',
                       boards: list = None) -> np.array:
    """
    :param dates:
    :param date_position_dic:
    :param code_order_dic:
    :param data_path:
    :param top_type: top类型
    :param boards: 板块
    :return:
    """
    top = np.zeros((len(dates), np.max(list(code_order_dic.values())) + 1), dtype=np.float32)
    univ = set(code_order_dic.keys())  # 所有股票univ
    if boards is None:
        boards = ['main_board']  # 默认主板股票
    for date in dates:
        with open('{}/StockDailyData/{}/stock.pkl'.format(data_path, date), 'rb') as f:
            stock = pickle.load(f)
        with open('{}/StockDailyData/{}/all_securities.pkl'.format(data_path, date), 'rb') as f:
            all_securities = pickle.load(f)
        print(f'{date} done.')
        with open('{}/StockDailyData/{}/st_dic.pkl'.format(data_path, date), 'rb') as f:
            st_dic = pickle.load(f)
        if top_type == 'listed':  # 当天仍在上市的股票
            for i in range(len(stock['code'])):
                if stock['code'][i] not in univ:
                    continue
                if stock['code'][i][:3] == '300':
                    if 'ChiNext' not in boards:  # 没有创业板就剔除创业板
                        continue
                elif stock['code'][i][:3] == '688':
                    if 'STAR' not in boards:  # 没有科创板就剔除科创板
                        continue
                elif 'main_board' not in boards:
                    continue
                if stock['paused'][i] == 1:
                    continue
                top[date_position_dic[date], code_order_dic[stock['code'][i]]] = 1
        for i in range(len(all_securities['code'])):
            if all_securities['code'][i] not in univ:
                continue
            if 'ST' in all_securities['display_name'][i]:
                top[date_position_dic[date], code_order_dic[all_securities['code'][i]]] = 0
            if st_dic[all_securities['code'][i]]:
                top[date_position_dic[date], code_order_dic[all_securities['code'][i]]] = 0
    top = top > 0
    return top


def generate_conbond_top(dates: np.array, date_position_dic: dict,
                         conbond_order_dic: dict, data_path: str, top_type: str = 'listed',
                         boards: list = None) -> np.array:  # 暂时top设置为当天在交易的可转债
    """
    :param dates:
    :param date_position_dic:
    :param conbond_order_dic:
    :param data_path:
    :param top_type: top类型
    :param boards: 板块
    :return:
    """
    top = np.zeros((len(dates), np.max(list(conbond_order_dic.values())) + 1), dtype=np.float32)
    univ = set(conbond_order_dic.keys())  # 所有可转债univ
    for date in dates:
        with open('{}/ConbondDailyData/{}/conbond.pkl'.format(data_path, date), 'rb') as f:
            conbond = pickle.load(f)
        print(f'{date} done.')
        if top_type == 'listed':  # 当天仍在上市的可转债
            for i in range(len(conbond['code'])):
                if conbond['code'][i] not in univ:
                    continue
                top[date_position_dic[date], conbond_order_dic[conbond['code'][i]]] = 1
    top = top > 0
    return top


def merge_top(top_list: list, method: str = 'cap') -> np.array:  # 多个top融合
    """
    :param top_list: top列表
    :param method: cap为取交集，cup为取并集
    :return:
    """
    top = top_list[0]
    if method == 'cap':
        for t in top_list[1:]:
            top = top & t
    elif method == 'cup':
        for t in top_list[1:]:
            top = top | t
    else:
        raise NotImplementedError
    return top
