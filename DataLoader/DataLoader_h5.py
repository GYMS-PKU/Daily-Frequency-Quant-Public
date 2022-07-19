# Copyright (c) 2022 Dai HBG

"""
h5 version
"""

import numpy as np
import jqdatasdk
import sys
import h5py


sys.path.append('C:/Users/18316/Desktop/Repositories/Daily-Frequency-Quant/DataLoader')
sys.path.append('C:/users/18316/Desktop/cy_lib')

from loaders.stock_daily_loader import *
from loaders.hf_data_loader import *
from loaders.conbond_daily_loader import *
from loaders.future_daily_loader import *
from loaders.index_daily_loader import *
from loaders.index_loader import *
from loaders.industry_loader import *
from loaders.future_hf_data_loader import *
from Data import *
from tools.trade_day import *
from data_processors.get_stock_daily_pv import *
from data_processors.get_conbond_daily_pv import *
from data_processors.get_future_daily_pv import *
from data_processors.get_stock_daily_money_flow import *
from data_processors.get_stock_daily_fundamental import *
from data_processors.get_stock_intra_pv import *
# from data_processors.get_future_intra_pv import *
from data_processors.get_industry_daily import *
from data_processors.get_stock_daily_ext import *
from data_processors.get_billboard import *
from tools.top_maker import *
from tools.calculate_tools import div_2d


class DataLoader_h5:  # 注意命名
    def __init__(self, user_id: str, password: str, data_path='D:/Documents/AutoFactoryData',
                 back_test_data_path='D:/Documents/AutoFactoryData/BackTestData', data_type: str = 'float32',
                 prediction_mode: bool = False, log_in: bool = False, stock_renew: bool = False):
        """
        :param user_id: 登录聚宽的用户id
        :param password: 登录密码
        :param data_path: 存放数据的路径
        :param back_test_data_path: 回测数据的存放路径
        :param prediction_mode: 是否是预测模式
        :param log_in: 是否登录
        :param stock_renew: 是否覆写股票日频行情
        """
        self.data_path = data_path
        self.back_test_data_path = back_test_data_path  # 该路径用于存放某一次回测所需要的任何字典
        self.user_id = user_id
        self.password = password
        if log_in:
            jqdatasdk.auth(self.user_id, self.password)  # 登录聚宽
        self.trade_day = TradeDay(data_path=data_path)  # 处理起止日期
        self.data_type = data_type
        self.prediction_mode = prediction_mode
        self.stock_renew = stock_renew

    """
    get_pv_data定义了从聚宽读取量价数据(pv for Price & Volume)并保存在本地的方法
    """

    def get_pv_data(self, start_date: str, end_date: str, data_type: list = None):  # 获得日频量价关系数据
        """
        :param start_date: 开始日期
        :param end_date: 结束日期，增量更新时这两个值设为相同
        :param data_type: 数据类型，stock_daily表示日频股票
        :return: 无返回值
        """
        if data_type is None:  # 参数默认值不要是可变的，否则可能出错
            data_type = ['stock_daily', 'industry']  # 默认获取股票日数据，行业和概念分类

        start_date = start_date.split('-')
        end_date = end_date.split('-')

        begin = datetime.date(int(start_date[0]), int(start_date[1]), int(start_date[2]))
        end = datetime.date(int(end_date[0]), int(end_date[1]), int(end_date[2]))
        start_date = self.trade_day.get_real_date(begin, direction='forward')  # 回测开始时间
        end_date = self.trade_day.get_real_date(end, direction='backward')  # 回测结束时间
        start_pos = self.trade_day.date_position_dic[start_date]
        end_pos = self.trade_day.date_position_dic[end_date]

        if 'stock_daily' in data_type:  # 获取日频量价、资金流数据
            get_stock_daily(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                            data_path=self.data_path, prediction_mode=self.prediction_mode,
                            stock_renew=self.stock_renew)

        if 'conbond_daily' in data_type:  # 获取可转债日频数据
            get_conbond_daily(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                              data_path=self.data_path, prediction_mode=self.prediction_mode)

        if 'future_daily' in data_type:  # 获取日频量价、资金流数据
            get_future_daily(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                             data_path=self.data_path, prediction_mode=self.prediction_mode)

        if 'index_daily' in data_type:  # 该字段获取的是所有指数数据
            get_index_daily(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                            data_path=self.data_path)

        if 'index' in data_type:  # 该字段获取的是中证500，中证1000，沪深300的成分股
            print('getting index data...')
            get_index(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                      data_path=self.data_path)

        if 'industry' in data_type:  # 获取行业分类，最后以字典形式存储
            get_industry_daily_data(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                    data_path=self.data_path)

        if '1m' in data_type:
            get_minute_data(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                            data_path=self.data_path, frequency='1m', prediction_mode=self.prediction_mode)

        if 'future_1m' in data_type:
            get_future_minute_data(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                   data_path=self.data_path, frequency='1m', prediction_mode=self.prediction_mode)

        if '10m' in data_type:
            get_minute_data(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                            data_path=self.data_path, frequency='10m', prediction_mode=self.prediction_mode)

    """
    get_matrix_data方法读取给定起始日期的原始数据，并生成需要的收益率矩阵，字典等

    v1.0
    2021-08-30
    - 需要解析一个字段，告诉DataLoader应该取从什么地方到什么地方的收益率，例如开盘价收益率或者日内收益率，周收益率等
        a. 具有形式"{}_{}_{}".format(data, data, day)的形式，表示从其中一个数据到另一个数据中间间隔day天
           例如open_close_4表示周一开盘价到周五收盘价
    2021-08-31
    1. 向前回溯例天数，默认100天，然后还要获得一个日期和下标对应的字典，以确定回测时的对应
    2021-09-02
    - get_matrix_data方法应当返回一个data，是一个Data类，包含了所需的调用信息，以后在各类之间交流信息更方便
    2021-09-04
    - get_matrix_data方法需要剔除科创版和创业板股票，并且返回Data类中需要写入top矩阵
    """

    def get_matrix_data(self, back_test_name: str = 'default',
                        start_date: str = '2021-01-01', end_date: str = '2021-06-30', back_windows: int = 10,
                        forward_windows: int = 10,
                        return_type: str = 'close-close-1',
                        fut_return_type: str = 'fut_open-fut_close-0',
                        data_type: list = None, univ: list = None, boards: list = None):
        # 该版本所有的数据都存为h5py
        """
        :param back_test_name: 文件存储命名
        :param start_date: 回测开始时间
        :param end_date: 结束时间
        :param back_windows: 开始时间向前多长的滑动窗口
        :param forward_windows: 开始时间向后多长的滑动窗口
        :param return_type: 该字段描述需要预测的收益率类型
        :param fut_return_type: 期货收益率
        :param data_type: 需要获得的数据类型
        :param univ: 使用哪些板块的股票作为univ
        :param boards: 使用哪些板块的股票构造top
        :return:
        """
        # 检查是否有缓存
        if '{}.pkl'.format(back_test_name) in os.listdir('{}/Cache'.format(self.data_path)):
            print('reading cache {}...'.format(back_test_name))
            with open('{}/Cache/{}.pkl'.format(self.data_path, back_test_name), 'rb') as f:
                data = pickle.load(f)
                return data

        tmp = start_date.split('-')
        start_date = self.trade_day.get_real_date(datetime.date(int(tmp[0]), int(tmp[1]), int(tmp[2])),
                                                  direction='forward')  # 回测开始时间
        tmp = end_date.split('-')
        end_date = self.trade_day.get_real_date(datetime.date(int(tmp[0]), int(tmp[1]), int(tmp[2])),
                                                direction='backward')  # 回测结束时间

        # 实际读取数据的起始和结束日期
        back_start_date = self.trade_day.back_search(start_date, back_windows=back_windows)
        # forward_end_date = self.trade_day.back_search(end_date, back_windows=-int(return_type.split('_')[-1]))
        forward_end_date = self.trade_day.back_search(end_date, back_windows=-forward_windows)  # 向后预留时间

        if data_type is None:
            # 默认读入股票纯日频量价数据和资金流数据
            data_type = ['stock_daily_pv', 'stock_daily_money_flow']

        # 生成日期位置字典
        start_pos = self.trade_day.date_position_dic[back_start_date]
        end_pos = self.trade_day.date_position_dic[forward_end_date]
        date_position_dic = {self.trade_day.trade_days[i]: i - start_pos for i in range(start_pos, end_pos + 1)}
        position_date_dic = {i - start_pos: self.trade_day.trade_days[i] for i in range(start_pos, end_pos + 1)}

        if 'future_daily_pv' in data_type:  # 生成期货品种和位置的字典
            print('generating futures set and dict...')
            dates = set(os.listdir('{}/FutureDailyData'.format(self.data_path)))  # 所有的日期
            future_set = set()  # 期货池
            for date in self.trade_day.trade_days[start_pos: end_pos + 1]:
                if str(date) not in dates:
                    print('date {} is empty, please load data first.'.format(date))
                    return
                with open('{}/FutureDailyData/{}/future.pkl'.format(self.data_path, date), 'rb') as f:
                    future_daily_data = pickle.load(f)
                futures = set([fut.split('.')[0][:-4] for fut in future_daily_data['code']])  # 期货只要品种名称
                future_set = future_set | futures  # 求并集
            future_set = sorted(list(future_set))  # 按照名称字典序排列

            # 生成期货和位置的映射字典
            future_order_dic = {future_set[i]: i for i in range(len(future_set))}
            order_future_dic = {i: future_set[i] for i in range(len(future_set))}
        else:
            future_order_dic = None
            order_future_dic = None

        if 'conbond_daily_pv' in data_type:  # 生成可转债对应的位置字典和正股的代码
            print('generating conbond set and dict...')
            dates = set(os.listdir('{}/ConbondDailyData'.format(self.data_path)))  # 所有的日期
            conbond_set = set()  # 可转债池
            conbond_stock_dic = {}  # 可转债到正股代码的映射
            for date in self.trade_day.trade_days[start_pos: end_pos + 1]:
                if str(date) not in dates:
                    print('date {} is empty, please load data first.'.format(date))
                    return
                with open('{}/ConbondDailyData/{}/conbond.pkl'.format(self.data_path, date), 'rb') as f:
                    conbond_daily_data = pickle.load(f)
                with open('{}/ConbondDailyData/{}/conbond_info.pkl'.format(self.data_path, date), 'rb') as f:
                    conbond_info = pickle.load(f)
                conbonds = set(conbond_daily_data['code'])
                for i in range(len(conbond_info['code'])):
                    conbond_stock_dic[conbond_info['code'][i]] = conbond_info['company_code'][i]
                conbond_set = conbond_set | conbonds  # 求并集
            conbond_set = sorted(list(conbond_set))  # 按照名称字典序排列

            # 生成可转债和位置的映射字典
            conbond_order_dic = {conbond_set[i]: i for i in range(len(conbond_set))}
            order_conbond_dic = {i: conbond_set[i] for i in range(len(conbond_set))}
            code_order_dic = {conbond_stock_dic[conbond_set[i]]: i for i in range(len(conbond_set))}
            order_code_dic = {i: conbond_stock_dic[conbond_set[i]] for i in range(len(conbond_set))}
        elif 'stock_daily_pv' in data_type:
            print('generating stocks set and dict...')
            dates = set(os.listdir('{}/StockDailyData'.format(self.data_path)))  # 所有的日期
            stock_set = set()  # 股票池
            for date in self.trade_day.trade_days[start_pos: end_pos + 1]:
                if str(date) not in dates:
                    print('date {} is empty, please load data first.'.format(date))
                    return
                with open('{}/StockDailyData/{}/stock.pkl'.format(self.data_path, date), 'rb') as f:
                    stock_daily_data = pickle.load(f)
                stocks = []
                for code in stock_daily_data['code']:
                    if code[:3] == '300':
                        if 'ChiNext' in univ:
                            stocks.append(code)
                    elif code[:3] == '688':
                        if 'STAR' in univ:
                            stocks.append(code)
                    elif 'main_board' in univ:
                        stocks.append(code)
                stocks = set(stocks)
                stock_set = stock_set | stocks  # 求并集
            stock_set = sorted(list(stock_set), key=lambda x: int(x[:6]))  # 按照代码从小到大排列
            # 生成股票和位置的映射字典
            code_order_dic = {stock_set[i]: i for i in range(len(stock_set))}
            order_code_dic = {i: stock_set[i] for i in range(len(stock_set))}
            conbond_order_dic = None
            order_conbond_dic = None
        else:
            conbond_order_dic = None
            order_conbond_dic = None
            code_order_dic = None
            order_code_dic = None

        # 首先读取股票或期货或可转债纯日频量价数据
        file_exist = False  # 记录文件是否存在
        if 'stock_daily_pv' in data_type:
            data_dic = get_stock_daily_pv(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                          date_position_dic=date_position_dic, code_order_dic=code_order_dic,
                                          data_path=self.data_path, data_type=self.data_type)
            if not file_exist:  # 如果文件不存在就新建
                writing_type = 'w'
                file_exist = True  # 文件已存在
            else:
                writing_type = 'a'
            f = h5py.File('{}/Cache_H5/{}.h5'.format(self.data_path, back_test_name), writing_type)
            for key, value in data_dic.items():
                f.create_dataset(key, data=value)
            f.close()
        if 'future_daily_pv' in data_type:
            data_dic = get_future_daily_pv(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                           date_position_dic=date_position_dic, future_order_dic=future_order_dic,
                                           data_path=self.data_path, data_type=self.data_type)
            if not file_exist:  # 如果文件不存在就新建
                writing_type = 'w'
                file_exist = True  # 文件已存在
            else:
                writing_type = 'a'
            f = h5py.File('{}/Cache_H5/{}.h5'.format(self.data_path, back_test_name), writing_type)
            for key, value in data_dic.items():
                f.create_dataset(key, data=value)
            f.close()
        if 'conbond_daily_pv' in data_type:  # 如果是可转债，那就默认把正股的股价也做了
            data_dic = get_stock_daily_pv(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                          date_position_dic=date_position_dic, code_order_dic=code_order_dic,
                                          data_path=self.data_path, data_type=self.data_type)
            conbond_data_dic = get_conbond_daily_pv(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                                    date_position_dic=date_position_dic,
                                                    conbond_order_dic=conbond_order_dic,
                                                    data_path=self.data_path, data_type=self.data_type)
            if not file_exist:  # 如果文件不存在就新建
                writing_type = 'w'
                file_exist = True  # 文件已存在
            else:
                writing_type = 'a'
            f = h5py.File('{}/Cache_H5/{}.h5'.format(self.data_path, back_test_name), writing_type)
            for key, value in data_dic.items():
                f.create_dataset(key, data=value)
            for key, value in conbond_data_dic.items():
                f.create_dataset(key, data=value)
            f.close()

        # 获取行业，形式是{'industry_name': np.array}，其中np.array使用数字标号，-1表示没有行业覆盖
        # 暂时所有数据都并列存，不需要新建group
        # 行业字典直接存在Data中，不另外写入
        if 'industry' in data_type:
            # 先获取行业字典
            industry_order_dic = {'swf': {}, 'sws': {}, 'swt': {}, 'concept': {}, 'jqf': {},
                                  'jqs': {}, 'zjw': {}}  # 行业编号到对应序号的字典，每个独立
            order_industry_dic = {'swf': {}, 'sws': {}, 'swt': {}, 'concept': {}, 'jqf': {},
                                  'jqs': {}, 'zjw': {}}  # 对应序号到行业编号的字典，每个独立
            ind_set_dic = {'swf': set(), 'sws': set(), 'swt': set(), 'concept': set(), 'jqf': set(),
                           'jqs': set(), 'zjw': set()}  # 记录已经出现过的行业编号的总数
            for date in self.trade_day.trade_days[start_pos: end_pos + 1]:
                # 前面已经检查过dates，因此这里不需要重复检查
                with open('{}/StockDailyData/{}/industry.pkl'.format(self.data_path, date), 'rb') as file:
                    tmp_industry_dic = pickle.load(file)
                    for ind_class, value in tmp_industry_dic.items():  # value也是一个字典，key是行业编号，value是股票列表
                        ind_set_dic[ind_class] = ind_set_dic[ind_class] | set(value.keys())
            for key in ind_set_dic.keys():
                ind_set_dic[key] = sorted(list(ind_set_dic[key]))  # 排序
                industry_order_dic[key] = {ind_set_dic[key][i]: i for i in range(len(ind_set_dic[key]))}
                order_industry_dic[key] = {i: ind_set_dic[key][i] for i in range(len(ind_set_dic[key]))}

            industry_dic = get_industry_daily(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                              date_position_dic=date_position_dic,
                                              code_order_dic=code_order_dic, industry_order_dic=industry_order_dic,
                                              data_path=self.data_path)
        else:
            industry_dic = None

        # 获得股票资金流数据
        if 'stock_daily_money_flow' in data_type:
            tmp_data_dic = get_stock_daily_money_flow(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                                      date_position_dic=date_position_dic,
                                                      code_order_dic=code_order_dic,
                                                      data_path=self.data_path, data_type=self.data_type)
            if not file_exist:  # 如果文件不存在就新建
                writing_type = 'w'
                file_exist = True  # 文件已存在
            else:
                writing_type = 'a'
            f = h5py.File('{}/Cache_H5/{}.h5'.format(self.data_path, back_test_name), writing_type)
            for key, value in tmp_data_dic.items():
                f.create_dataset(key, data=value)  # 文件类型是float32
            f.close()

        # 获得股票基本面数据
        if 'stock_daily_fundamental' in data_type:
            tmp_data_dic = get_stock_daily_fundamental(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                                       date_position_dic=date_position_dic,
                                                       code_order_dic=code_order_dic,
                                                       data_path=self.data_path, data_type=self.data_type)
            if not file_exist:  # 如果文件不存在就新建
                writing_type = 'w'
                file_exist = True  # 文件已存在
            else:
                writing_type = 'a'
            f = h5py.File('{}/Cache_H5/{}.h5'.format(self.data_path, back_test_name), writing_type)
            for key, value in tmp_data_dic.items():
                f.create_dataset(key, data=value)  # 文件类型是float32
            f.close()

        # 获得股票拓展数据
        if 'stock_daily_ext' in data_type:
            tmp_data_dic = get_stock_daily_ext(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                               date_position_dic=date_position_dic,
                                               code_order_dic=code_order_dic,
                                               data_path=self.data_path, data_type=self.data_type)
            if not file_exist:  # 如果文件不存在就新建
                writing_type = 'w'
                file_exist = True  # 文件已存在
            else:
                writing_type = 'a'
            f = h5py.File('{}/Cache_H5/{}.h5'.format(self.data_path, back_test_name), writing_type)
            for key, value in tmp_data_dic.items():
                f.create_dataset(key, data=value)  # 文件类型是float32
            f.close()

        # 获得股票龙虎榜数据
        if 'billboard' in data_type:
            tmp_data_dic = get_billboard(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                         date_position_dic=date_position_dic,
                                         code_order_dic=code_order_dic,
                                         data_path=self.data_path, data_type=self.data_type)
            if not file_exist:  # 如果文件不存在就新建
                writing_type = 'w'
                file_exist = True  # 文件已存在
            else:
                writing_type = 'a'
            f = h5py.File('{}/Cache_H5/{}.h5'.format(self.data_path, back_test_name), writing_type)
            for key, value in tmp_data_dic.items():
                f.create_dataset(key, data=value)  # 文件类型是float32
            f.close()

        # 目前实现三种频率的分钟数据，需要测试1min的读取速度以及内存暂用，应该需要改为单精度存储
        if '1m' in data_type:
            tmp_data_dic = get_stock_intra_pv(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                              date_position_dic=date_position_dic,
                                              code_order_dic=code_order_dic, frequency=1,
                                              data_path=self.data_path, data_type=self.data_type)
            if not file_exist:  # 如果文件不存在就新建
                writing_type = 'w'
                file_exist = True  # 文件已存在
            else:
                writing_type = 'a'
            f = h5py.File('{}/Cache_H5/{}.h5'.format(self.data_path, back_test_name), writing_type)
            for key, value in tmp_data_dic.items():
                f.create_dataset(key, data=value)  # 文件类型是float32
            f.close()

        if '10m' in data_type:
            tmp_data_dic = get_stock_intra_pv(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                              date_position_dic=date_position_dic,
                                              code_order_dic=code_order_dic, frequency=10,
                                              data_path=self.data_path, data_type=self.data_type)
            if not file_exist:  # 如果文件不存在就新建
                writing_type = 'w'
                file_exist = True  # 文件已存在
            else:
                writing_type = 'a'
            f = h5py.File('{}/Cache_H5/{}.h5'.format(self.data_path, back_test_name), writing_type)
            for key, value in tmp_data_dic.items():
                f.create_dataset(key, data=value)  # 文件类型是float32
            f.close()

        if '5m' in data_type:
            tmp_data_dic = get_stock_intra_pv(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                              date_position_dic=date_position_dic,
                                              code_order_dic=code_order_dic, frequency=5,
                                              data_path=self.data_path, data_type=self.data_type)
            if not file_exist:  # 如果文件不存在就新建
                writing_type = 'w'
                file_exist = True  # 文件已存在
            else:
                writing_type = 'a'
            f = h5py.File('{}/Cache_H5/{}.h5'.format(self.data_path, back_test_name), writing_type)
            for key, value in tmp_data_dic.items():
                f.create_dataset(key, data=value)  # 文件类型是float32
            f.close()

        # 生成top
        if 'stock_daily_pv' in data_type:
            print('getting top')
            if boards is None:  # 默认主板股票
                boards = ['main_board']
            top = generate_stock_top(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                     date_position_dic=date_position_dic,
                                     code_order_dic=code_order_dic, data_path=self.data_path, top_type='listed',
                                     boards=boards)

            f = h5py.File('{}/Cache_H5/{}.h5'.format(self.data_path, back_test_name), 'a')
            f.create_dataset('top', data=top)
            f.create_dataset('ret', data=np.zeros(top.shape, dtype=np.float32))
            f.create_dataset('zdt_top', data=np.zeros(top.shape, dtype=np.float32))
            f.close()
            conbond_top = None
        elif 'conbond_daily_pv' in data_type:
            print('getting top')
            if boards is None:  # 默认主板股票
                boards = ['main_board']
            top = generate_stock_top(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                     date_position_dic=date_position_dic,
                                     code_order_dic=code_order_dic, data_path=self.data_path, top_type='listed',
                                     boards=boards)
            conbond_top = generate_conbond_top(dates=self.trade_day.trade_days[start_pos: end_pos + 1],
                                               date_position_dic=date_position_dic,
                                               conbond_order_dic=conbond_order_dic, data_path=self.data_path,
                                               top_type='listed',
                                               boards=boards)

            f = h5py.File('{}/Cache_H5/{}.h5'.format(self.data_path, back_test_name), 'a')
            f.create_dataset('top', data=top)
            f.create_dataset('conbond_top', data=conbond_top)
            f.create_dataset('ret', data=np.zeros(top.shape, dtype=np.float32))
            f.create_dataset('zdt_top', data=np.zeros(top.shape, dtype=np.float32))
            f.close()
        else:
            conbond_top = None

        # 生成Data
        data = Data(code_order_dic=code_order_dic, order_code_dic=order_code_dic, future_order_dic=future_order_dic,
                    order_future_dic=order_future_dic, conbond_order_dic=conbond_order_dic,
                    order_conbond_dic=order_conbond_dic,
                    date_position_dic=date_position_dic, position_date_dic=position_date_dic,
                    data_dic=None, ret=np.zeros(top.shape, dtype=np.float32),
                    trade_day=self.trade_day, industry_dic=industry_dic,
                    top_dic=None, top=top, conbond_top=conbond_top, zdt_top=None,
                    data_path='{}/Cache_H5/{}.h5'.format(self.data_path, back_test_name),
                    raw_data_path=self.data_path, start_pos=start_pos, end_pos=end_pos)
        data.set_ret(return_type, 'float32')

        # 写入数据，直接写入Data
        with open('{}/Cache/{}.pkl'.format(self.data_path, back_test_name), 'wb') as f:
            pickle.dump(data, f)
        return data
