# -*- coding: utf-8 -*-

"""
Description:
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import global_list as gl
from utils import preproce_data
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_forex_data(data_name):
    # data_name = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'XAUUSD']

    data = pd.read_csv(gl.TEST_FOREX_RESULT_PATH + data_name + '1440-d_form.csv')
    print('get data ' + data_name)

def show_history_of_close():
    forex_close = pd.DataFrame({})
    forex_p_change = pd.DataFrame({})
    data_name = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'XAUUSD']
    for name in data_name:
        data = pd.read_csv(gl.TEST_FOREX_RESULT_PATH + name + '1440-d_form.csv')[1:]

        #print(data[:3])
        #data = preproce_data.PreproceData().foredata_to_day_form(data['fclose'])
        print('get data ' + name)
        #print(len(data))

        scaler = MinMaxScaler(feature_range=(0, 1))
        #forex_close[name] = scaler.fit_transform(data['fclose'].values.reshape(-1,1))
        close_data = np.array([data['fclose']]).reshape(1,-1)
        p_change_data = np.array([data['p_change']]).reshape(1,-1)
        forex_close[name] = scaler.fit_transform(close_data)
        forex_p_change[name] = scaler.fit_transform(p_change_data)

        # forex_close[name] = data['fclose']
        # forex_p_change[name] = data['p_change']

    forex_close.to_csv(gl.TEST_FOREX_RESULT_PATH + 'forex_close.csv')
    forex_p_change.to_csv(gl.TEST_FOREX_RESULT_PATH + 'forex_p_change.csv')
    print(forex_close[:3])
    print(forex_p_change[:3])

def show_history_of_close2():
    data_name = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'XAUUSD']
    pic_num = 330
    for name in data_name:
        data = pd.read_csv(gl.TEST_FOREX_RESULT_PATH + name + '1440-d_form.csv')
        pic_num = pic_num + 1
        plt.subplot(pic_num)
        plt.plot(data['fclose'])
        plt.title(name)

    plt.show()

def show_history_of_close3():
    data_name = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'XAUUSD']
    pic_num = 330
    for name in data_name:
        data = pd.read_csv(gl.TEST_FOREX_RESULT_PATH + name + '1440-d_form.csv')
        pic_num = pic_num + 1
        plt.subplot(pic_num)
        data['fclose'].plot(label=name)
        plt.legend(loc='best')

    plt.show()


def plot_p_change_heatmap():
    data = pd.read_csv(gl.TEST_FOREX_RESULT_PATH + 'forex_p_change.csv')
    print(data[:3].ix[:,1:9])
    sns.heatmap(data.ix[:,1:9].corr(), annot=True, cmap='RdYlGn', linewidths=0.1, vmin=0)
    plt.show()

    data_name = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'XAUUSD']
    for name in data_name:
        print(name)
        print(data[name].describe())


def show_close_ma(ma_fast, ma_slow):
    data_name = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'XAUUSD']
    pic_num = 330
    for name in data_name:
        data = pd.read_csv(gl.TEST_FOREX_RESULT_PATH + name + '1440-d_form.csv')
        pic_num = pic_num + 1
        plt.subplot(pic_num)
        data['fast'] = data['fclose'].rolling(window=ma_fast).mean()
        data['slow'] = data['fclose'].rolling(window=ma_slow).mean()
        data['fclose'].plot(label=name)
        data['fast'].plot(label=ma_fast)
        data['slow'].plot(label=ma_slow)
        plt.legend(loc='best')

    plt.show()




# Simpe test
#get_forex_data('AUDUSD')
#show_history_of_close()
#show_history_of_close3()
#plot_p_change_heatmap()
show_close_ma(240, 360)