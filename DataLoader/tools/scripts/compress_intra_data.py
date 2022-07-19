# Copyright (c) 2022 Dai HBG


"""
该脚本用于将1分钟数据中不在市的部分删除
"""


import os
import pandas as pd


def main():
    data_path = 'D:/Documents/AutoFactoryData'
    intra_path = 'E:/Backups/AutoFactoryData/StockIntraDayData/1m'
    dates = os.listdir(intra_path)
    for date in dates:
        all_securities = pd.read_csv('{}/StockDailyData/{}/all_securities.csv'.format(data_path, date))
        all_stocks = set(all_securities['code'])
        files = os.listdir('{}/{}'.format(intra_path, date))
        for f in files:
            if f[:-4] not in all_stocks:
                os.remove('{}/{}/{}'.format(intra_path, date, f))
        print('{} done.'.format(date))


if __name__ == '__main__':
    main()
