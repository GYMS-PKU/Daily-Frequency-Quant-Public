# Copyright (c) 2022 Dai HBG


"""
本代码定义一个用于管理交易日的类

日志
2022-01-07
- init
"""


import pickle
import datetime


class TradeDay:
    def __init__(self, data_path: str = 'D:/Documents/AutoFactoryData'):
        self.data_path = data_path
        with open('{}/BaseData/trade_days.pkl'.format(data_path), 'rb') as f:
            self.trade_days = pickle.load(f)

        # 建立位置和日期的映射
        self.position_date_dic = {i: self.trade_days[i] for i in range(len(self.trade_days))}
        self.date_position_dic = {self.trade_days[i]: i for i in range(len(self.trade_days))}

    # 获得向前或者向后的最靠近的交易日
    def get_real_date(self, date: datetime.date, direction: str = 'forward') -> datetime.date:
        """
        :param date: 日期
        :param direction: forward指向后搜索，backward指向前搜索
        :return:
        """
        if direction not in ['forward', 'backward']:
            raise NotImplementedError('direction should be forward or backward.')
        try:
            _ = self.date_position_dic[date]
            return date
        except KeyError:
            for i in range(len(self.trade_days)):
                if date < self.trade_days[i]:  # 前一个位置是最最靠近的
                    if direction == 'backward':  # 向前搜索
                        if i == 0:
                            print('warning: earliest trade day is {}.'.format(self.trade_days[0]))
                            return self.trade_days[0]
                        return self.trade_days[i - 1]
                    elif direction == 'forward':  # 向后搜索
                        return self.trade_days[i]
            print('warning: latest trade day is {}.'.format(self.trade_days[-1]))
            return self.trade_days[-1]  # 右边溢出

    def back_search(self, date: datetime.date, back_windows: int):  # 当前交易日回溯若干交易日
        """
        :param date: 日期
        :param back_windows: 回溯的天数
        :return:
        """
        try:
            pos = self.date_position_dic[date]
        except KeyError:
            print('date should be a trade day, please call get_real_date to get a trade day.')
            return
        new_pos = max(0, pos - back_windows)
        return self.position_date_dic[new_pos]
