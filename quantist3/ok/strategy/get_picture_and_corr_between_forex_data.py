# -*- coding: utf-8 -*-

"""
Description:
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import global_list as gl

def get_forex_data(data_name):
    # data_name = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'XAUUSD']

    data = pd.read_csv(gl.TEST_FOREX_RESULT_PATH + data_name + '1440-d_form.csv')
    print('get data ' + data_name)

def show_history_of_close():
    data_name = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'XAUUSD']
    for name in data_name:
        data = pd.read_csv(gl.TEST_FOREX_RESULT_PATH + name + '1440-d_form.csv')
        #print('get data ' + name)


# Simpe test
#get_forex_data('AUDUSD')