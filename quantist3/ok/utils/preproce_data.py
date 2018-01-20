# -*- coding: utf-8 -*-

# quantist
# 
# Copyright 2017-2018 Pan Liu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
# by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Unless required

""" 
Author: liupan 
"""

"""
Description:
"""
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import global_list as gl

"""
# data_path = "../data/pool/"
# result_path = "../data/result/"
# 
# data = pd.read_excel(data_path + "Google_Stock_Price_Test.xls")
# 
# print(data['close'][:5])

# 0    3297.06
# 1    3300.06
# 2    3287.61
# 3    3296.54
# 4    3267.92
# Name: close, dtype: float64

# print("--------------------------------------")
# values = data['close'].values.reshape(-1,1)
# print(values[:5])

# [[ 3297.06]
#  [ 3300.06]
#  [ 3287.61]
#  [ 3296.54]
#  [ 3267.92]]

# print("--------------------------------------")
# values = values.astype('float64')
# print(values[:5])
# """
# [[ 3297.06]
#  [ 3300.06]
#  [ 3287.61]
#  [ 3296.54]
#  [ 3267.92]]

# print("--------------------------------------")
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# print(scaled[:5])

# [[ 0.25546762]
#  [ 0.25666251]
#  [ 0.25170371]
#  [ 0.25526051]
#  [ 0.24386125]]

# print("--------------------------------------")
#
# sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.1, vmin=0)
# plt.show()s

class PreproceData():
    """
    Change the data from diffrent source to my from for day, tick, or found and so on.

    For day
        csv: 'num','data','open','high','low','close','adj','volumn','change'
        db: 't_num','t_data','t_high'...
        'adj': for the data from other resource, such as yahool
        'change':change = today_close - yesterday_close
    """
    def to_day_form(data):
        """
        Tushare data to day form
        :return:
        """
        if (data.empty):
            print("Data is None")
            return None
        else:
            data.reset_index(inplace=True)
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
            return data

    def foredata_to_day_form(self, data):
        """
        Forex data to day form
        :return:
        """
        if (data.empty):
            print("Data is None")
            return None
        else:
            #data.reset_index(inplace=True)
            print(str(data['fdated']).strftime("%Y-%m-%d"))
            data['fdated'] = pd.to_datetime(data['fdated'])
            data = data.set_index('fdated')
            return data


    def to_json(data):
        """
        # Change the form from DataForm to json
        # ready to complate...
        :return:
        """
        pass

    def forex_basic(data):
        if (data.empty):
            print("Data is None")
            return None
        else:
            data.reset_index(inplace=True)
            data['fdated'] = pd.to_datetime(data['fdated'])
            data = data.set_index('fdated')
            return data

    def forex_to_standard(self, data, save_name, save_path, saved=False):
        """
        Change the original data to standard form
        original data form:
            ftimed,ftimeh,fopen,fhigh,flow,fclose,fvolume
            2010.11.05,00:00,1391.40,1398.15,1373.50,1395.30,4541
            2010.11.07,00:00,1395.40,1395.70,1393.00,1395.45,536
        :return:
        """

        data_len = len(data['fclose'])
        print('The length of data is ' + str(data_len))
        data['ma5'] = data['fclose'].rolling(window=5).mean()
        data['price_change'] = data['fclose'].diff()
        data['shift'] = data['fclose'].shift()
        data['p_change'] = data['price_change']*100/data['shift']
        print(data[:8])
        if saved==True:
            data.to_csv(save_path + save_name)
            print('Save successed')
        else:
            print('No saved')


    def test_gg(self):
        original_data = pd.read_csv(gl.TEST_FOREX_XAUUSDD_DATA)
        result_data_path = gl.TEST_RESULT_POOL_PATH
        original_data['shift'] = original_data['fclose'].shift()
        print(original_data[:8])


# Simple test

#Test forex_to_standard

"""
# data_name = 'XAUUSD-d'
# data = pd.read_csv(gl.TEST_FOREX_DATA_PATH + data_name + '.csv')
# save_name = data_name + '_form.csv'
# result_data_path = gl.TEST_FOREX_RESULT_PATH
# p = PreproceData()
# p.forex_to_standard(data, save_name, result_data_path, saved=True)
"""


#f = p.test_gg()


