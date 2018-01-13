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
Create date: 
"""

"""
Description:
"""
import pandas as pd
"""
Change the data from diffrent source to my from for day, tick, or found and so on.

For day
    csv: 'num','data','open','high','low','close','adj','volumn','change'
    db: 't_num','t_data','t_high'...
    'adj': for the data from other resource, such as yahool
    'change':change = today_close - yesterday_close
"""
class ChangeDataForm(object):

    # Get the data from tushare or DataForme form.
    def to_day_form(data):
        if (data.empty):
            print("Data is None")
            return None
        else:
            data.reset_index(inplace=True)
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
            return data

    # Change the form from DataForm to json
    # ready to complate...
    def to_json(data):
        pass