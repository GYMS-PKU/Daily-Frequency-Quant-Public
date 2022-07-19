# Copyright (c) 2021-2022 Dai HBG


"""
该文件定义了所有默认算子字典
"""

default_operation_dic = {'1': ['csrank', 'zscore', 'neg', 'csindneutral', 'csindneutral', 'csindmean',
                               'csindsum',
                               'csindmax', 'csindmin', 'csindstd', 'absv', 'expv', 'logv',
                               'csmean'],

                         '2': ['add', 'prod', 'minus', 'div', 'lt', 'le', 'gt', 'ge', 'intraweightedavg',
                               'intraregres', 'intraregalpha', 'intraregbeta'],
                         '2_num': ['tscorr', 'tscov', 'tsregres', 'tsregalpha', 'tsregbeta', 'c_topn'],

                         'intra_data': ['intra_open', 'intra_high', 'intra_low',
                                        'intra_close', 'intra_avg', 'intra_volume',
                                        'intra_money']
                         }

default_dim_operation_dic = {'2_2': ['csrank', 'zscore', 'neg', 'csindneutral', 'csind', 'absv',
                                     'wdirect', 'tsrank', 'tskurtosis', 'tsskew',
                                     'tsmean', 'tsstd', 'tsdelay', 'tsdelta', 'tsmax',

                                     'discrete', 'log', 'logv',
                                     ],
                             '3_3': ['neg', 'absv', 'add', 'prod', 'minus', 'div', 'lt', 'le', 'gt', 'ge',
                                     'intratsregres', 'log', 'logv',

                                     'biintratsquantiledownmean'],
                             '3_2': ['intratsmax', 'intratsmaxpos', 'intratsmin',

                                     'biintraquantile', 'biintraquantileupmean', 'biintraquantiledownmean']
                             }
