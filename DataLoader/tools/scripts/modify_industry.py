# Copyright (c) 2022 Dai HBG


"""
该脚本用于将industry数据改名
"""


import os
import pickle


def main():
    data_path = 'D:/Documents/AutoFactoryData/StockDailyData'
    dates = os.listdir(data_path)
    for date in dates:
        if 'industry.pkl' in os.listdir('{}/{}'.format(data_path, date)):
            if 'industry_{}.pkl'.format(date) in os.listdir('{}/{}'.format(data_path, date)):
                os.remove('{}/{}/industry_{}.pkl'.format(data_path, date, date))
        else:
            with open('{}/{}/industry_{}.pkl'.format(data_path, date, date), 'rb') as f:
                industry = pickle.load(f)
            with open('{}/{}/industry.pkl'.format(data_path, date), 'wb') as f:
                pickle.dump(industry, f)
            os.remove('{}/{}/industry_{}.pkl'.format(data_path, date, date))
        print('{} done.'.format(date))


if __name__ == '__main__':
    main()
