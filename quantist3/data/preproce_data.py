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

data_path = "../data/pool/"
result_path = "../data/result/"

data = pd.read_excel(data_path + "Google_Stock_Price_Test.xls")

print(data['close'][:5])
"""
0    3297.06
1    3300.06
2    3287.61
3    3296.54
4    3267.92
Name: close, dtype: float64
"""
print("--------------------------------------")
values = data['close'].values.reshape(-1,1)
print(values[:5])
"""
[[ 3297.06]
 [ 3300.06]
 [ 3287.61]
 [ 3296.54]
 [ 3267.92]]
"""
print("--------------------------------------")
values = values.astype('float64')
print(values[:5])
"""
[[ 3297.06]
 [ 3300.06]
 [ 3287.61]
 [ 3296.54]
 [ 3267.92]]
"""
print("--------------------------------------")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print(scaled[:5])
"""
[[ 0.25546762]
 [ 0.25666251]
 [ 0.25170371]
 [ 0.25526051]
 [ 0.24386125]]
"""
print("--------------------------------------")

sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.1, vmin=0)
plt.show()