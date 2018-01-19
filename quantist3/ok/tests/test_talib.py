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

import tushare as ts
import talib


df = ts.get_hist_data('sh')
closed = df['close'].values

ccitalib=talib.CCI(df['high'].values, df['low'].values, df['close'].values,timeperiod=10)[-1]
print(ccitalib) # -106.212259763

# ma5 = talib.MA(df['close'], )

cmo = talib.CMO(df['close'].values, timeperiod=14)
# print(cmo)

ma3 = talib.SMA(closed, timeperiod=3)
print(ma3[:5]) # [           nan            nan  3285.33333333  3289.89666667  3302.42      ]








