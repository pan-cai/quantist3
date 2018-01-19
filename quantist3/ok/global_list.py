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

import pandas as pd

TEST_DATA_POOL_PATH = '/home/liupan/waihui/quantist4/data/pool/'

TEST_RESULT_POOL_PATH = '/home/liupan/waihui/quantist4/data/result/'

TEST_FOREX_DATA_PATH = '/home/liupan/waihui/quantist4/data/pool/forex/data/'
TEST_FOREX_RESULT_PATH = '/home/liupan/waihui/quantist4/data/pool/forex/result/'

#TEST_SH_PRICES = pd.read_excel(TEST_DATA_POOL_PATH + 'sh2.xls').sort_index(ascending=False)

TEST_FOREX_XAUUSDD_DATA = TEST_FOREX_DATA_PATH + 'data/XAUUSD-d.csv'

# print(TEST_CLOSE_PRICES)