# Copyright (c) 2022 Dai HBG


import numpy as np
import datetime
import h5py
import sys

sys.path.append('C:/Users/Administrator/Desktop/Daily-Frequency-Quant/QBG/DataLoader')
sys.path.append('C:/Users/HBG/Desktop/Repositories/Daily-Frequency-Quant/QBG/DataLoader')
from tools.trade_day import *
from tools.calculate_tools import *
from tools.top_maker import *


class Data:
    def __init__(self, code_order_dic: dict = None, order_code_dic: dict = None, future_order_dic: dict = None,
                 order_future_dic: dict = None, conbond_order_dic: dict = None,
                 order_conbond_dic: dict = None, date_position_dic: dict = None, position_date_dic: dict = None,
                 data_dic: dict = None, ret: np.array = None, fut_ret: np.array = None, conbond_ret: np.array = None,
                 trade_day: TradeDay = None, start_pos: int = 0, end_pos: int = 1000, industry_dic: dict = None,
                 top_dic: dict = None, top: np.array = None, fut_top: np.array = None, conbond_top: np.array = None,
                 zdt_top: np.array = None,
                 dt_top: np.array = None, zt_top: np.array = None, fut_zdt_top: np.array = None,
                 fut_dt_top: np.array = None,
                 fut_zt_top: np.array = None, data_path: str = None,
                 raw_data_path: str = None):
        """
        :param code_order_dic: 股票代码到矩阵位置的字典
        :param order_code_dic: 矩阵位置到股票代码的字典
        :param future_order_dic: 期货代码到矩阵位置的字典
        :param order_future_dic: 矩阵位置到期货代码的字典
        :param conbond_order_dic: 可转债代码到矩阵位置的字典
        :param order_conbond_dic: 矩阵位置到可转债代码的字典
        :param date_position_dic: 日期到矩阵下标的字典
        :param data_dic: 所有的数据，形状一致
        :param ret: 股票收益率
        :param fut_ret: 期货收益率
        :param conbond_ret: 债券收益率
        :param trade_day: TradeDay类
        :param start_pos: univ开始下标
        :param end_pos: univ结束下标
        :param industry_dic: 使用的行业分类，是一个字典，值是一个矩阵，里面的数字就是行业分类
        :param top_dic: 字典，存储top矩阵中存储每一个交易日可选的股票
        :param top: 股票top
        :param fut_top: 期货top
        :param conbond_top: 可转债top
        :param zdt_top：涨跌停top
        :param dt_top：跌停top
        :param zt_top：涨停top
        :param data_path: 如果是h5模式则需要传入路径名
        :param raw_data_path: 原始数据的路径
        """
        self.code_order_dic = code_order_dic
        self.order_code_dic = order_code_dic
        self.future_order_dic = future_order_dic
        self.order_future_dic = order_future_dic
        self.conbond_order_dic = conbond_order_dic
        self.order_conbond_dic = order_conbond_dic
        self.date_position_dic = date_position_dic
        self.position_date_dic = position_date_dic
        self.data_dic = data_dic
        self.ret = ret
        self.fut_ret = fut_ret
        self.conbond_ret = conbond_ret
        self.industry_dic = industry_dic
        self.trade_day = trade_day
        self.start_pos = start_pos
        self.end_pos = end_pos
        if top_dic is None:
            top_dic = {}
        self.top_dic = top_dic
        self.top = top
        self.conbond_top = conbond_top
        self.fut_top = fut_top
        # 股票涨跌停
        self.zdt_top = zdt_top
        self.dt_top = dt_top
        self.zt_top = zt_top
        # 期货涨跌停
        self.fut_zdt_top = fut_zdt_top
        self.fut_dt_top = fut_dt_top
        self.fut_zt_top = fut_zt_top
        self.data_path = data_path
        if data_path is None:
            self.data_mode = 'dic'
        else:
            self.data_mode = 'h5'

        self.raw_data_path = raw_data_path  # 原始数据路径

        self.industry = None  # 使用的行业
        self.max_ind_code = 0  # 行业中最大的字段编号

    def set_top(self, boards: list):  # 该方法目前只支持股票
        if boards[0] not in self.top_dic.keys():
            top = generate_stock_top(dates=self.trade_day.trade_days[self.start_pos: self.end_pos + 1],
                                     date_position_dic=self.date_position_dic,
                                     code_order_dic=self.code_order_dic, data_path=self.raw_data_path,
                                     top_type='listed', boards=boards[:1])
            self.top_dic[boards[0]] = top
        top = self.top_dic[boards[0]]
        for i in range(1, len(boards)):
            if boards[i] not in self.top_dic.keys():
                top_tmp = generate_stock_top(dates=self.trade_day.trade_days[self.start_pos: self.end_pos + 1],
                                             date_position_dic=self.date_position_dic,
                                             code_order_dic=self.code_order_dic, data_path=self.raw_data_path,
                                             top_type='listed', boards=boards[i: i + 1])
                self.top_dic[boards[i]] = top_tmp
            else:
                top_tmp = self.top_dic[boards[i]]
            top = top | top_tmp
        self.top = top

    def set_ret(self, return_type: str, data_type: str = 'float32', sec: str = 'stock'):  # 同时更改ret和zdt_top
        length = int(return_type.split('-')[-1])
        start_name = return_type.split('-')[0]
        end_name = return_type.split('-')[1]
        if data_type == 'float64':  # 2022-05-15: float64废弃了，以后去掉
            ret = np.zeros(self.ret.shape)
            ret[-length:] = np.nan  # 注意要将不合法的部分全部置为nan
            ret[:-length] = div_2d(self.data_dic[end_name][length:], self.data_dic[start_name][:-length]) - 1
        elif data_type == 'float32':
            ret = np.zeros(self.ret.shape, dtype=np.float32)
            if length >= 1:
                ret[-length:] = np.nan  # 注意要将不合法的部分全部置为nan
            if self.data_mode == 'dic':
                if length >= 1:
                    ret[:-length] = div_2d_32(self.data_dic[end_name][length:], self.data_dic[start_name][:-length]) - 1
                else:
                    ret = div_2d_32(self.data_dic[end_name], self.data_dic[start_name]) - 1
            elif self.data_mode == 'h5':
                if length >= 1:
                    ret[:-length] = div_2d_32(h5py.File(self.data_path, 'r')[end_name][length:],
                                              h5py.File(self.data_path, 'r')[start_name][:-length]) - 1
                else:
                    ret = div_2d_32(h5py.File(self.data_path, 'r')[end_name][:],
                                    h5py.File(self.data_path, 'r')[start_name][:]) - 1
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        if sec == 'stock':
            self.ret = ret
        elif sec == 'future':
            self.fut_ret = ret
        elif sec == 'conbond':
            self.conbond_ret = ret

        # 用涨跌停价格判断涨跌停
        if sec == 'stock':
            if self.data_mode == 'dic':
                zt_top = np.zeros(self.ret.shape, dtype=np.float32)
                zt_top[np.isnan(self.data_dic['high_limit'])] = 0
                zt_top = zt_top > 0
                se = (~np.isnan(self.data_dic['high_limit']))
                zt_top[se] = (self.data_dic[start_name][se] < self.data_dic['high_limit'][se])
                zt_top[:-1] = zt_top[1:]
                zt_top[-1] = False

                dt_top = np.zeros(self.ret.shape, dtype=np.float32)
                dt_top[np.isnan(self.data_dic['low_limit'])] = 0
                dt_top = dt_top > 0
                se = (~np.isnan(self.data_dic['low_limit']))
                dt_top[se] = (self.data_dic[start_name][se] > self.data_dic['low_limit'][se])
                dt_top[:-1] = dt_top[1:]
                dt_top[-1] = False

            elif self.data_mode == 'h5':
                zt_top = np.zeros(self.ret.shape, dtype=np.float32)
                zt_top[np.isnan(h5py.File(self.data_path, 'r')['high_limit'])] = 0
                zt_top = zt_top > 0
                se = (~np.isnan(h5py.File(self.data_path, 'r')['high_limit']))
                zt_top[se] = (h5py.File(self.data_path, 'r')[start_name][se] < h5py.File(self.data_path,
                                                                                         'r')['high_limit'][se])
                zt_top[:-1] = zt_top[1:]
                zt_top[-1] = False

                dt_top = np.zeros(self.ret.shape, dtype=np.float32)
                dt_top[np.isnan(h5py.File(self.data_path, 'r')['low_limit'])] = 0
                dt_top = dt_top > 0
                se = (~np.isnan(h5py.File(self.data_path, 'r')['low_limit']))
                dt_top[se] = (h5py.File(self.data_path, 'r')[start_name][se] > h5py.File(self.data_path,
                                                                                         'r')['low_limit'][se])
                dt_top[:-1] = dt_top[1:]
                dt_top[-1] = False
            else:
                raise NotImplementedError
            self.zdt_top = zt_top & dt_top
            self.zt_top = zt_top
            self.dt_top = dt_top
        elif sec == 'future':
            if self.data_mode == 'dic':
                self.fut_zt_top = (self.data_dic[start_name] < self.data_dic['high_limit']) & \
                                  (~np.isnan(self.data_dic[start_name]))
                self.fut_dt_top = (self.data_dic[start_name] > self.data_dic['low_limit']) & \
                                  (~np.isnan(self.data_dic[start_name]))
                self.fut_zdt_top = self.fut_zt_top & self.fut_dt_top
            elif self.data_mode == 'h5':
                self.fut_zt_top = (h5py.File(self.data_path, 'r')[start_name] < h5py.File(self.data_path,
                                                                                          'r')['high_limit']) & \
                                  (~np.isnan(h5py.File(self.data_path, 'r')[start_name]))
                self.fut_dt_top = (h5py.File(self.data_path, 'r')[start_name] > h5py.File(self.data_path,
                                                                                          'r')['low_limit']) & \
                                  (~np.isnan(h5py.File(self.data_path, 'r')[start_name]))
                self.fut_zdt_top = self.fut_zt_top & self.fut_dt_top
            else:
                raise NotImplementedError

    def set_ind(self, name: str = 'sws'):
        self.industry = self.industry_dic[name]
        self.max_ind_code = np.nanmax(self.industry)

    def get_real_date(self, date: str, direction: str = 'forward') -> int:  # 用于获取起始日期对应的真正的数据起始位置
        """
        :param date: 任意输入的结束日期
        :param direction: 方向
        :return: 返回有交易的真正的起始日期对应的下标
        """
        tmp_date = date.split('-')
        real_date = self.trade_day.get_real_date(date=datetime.date(int(tmp_date[0]), int(tmp_date[1]),
                                                                    int(tmp_date[2])), direction=direction)
        return self.date_position_dic[real_date]
