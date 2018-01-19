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
# _*_ coding:utf-8 _*_
"""
获取当天所有的数据，并存入excel中
"""

import tushare as ts
import pandas as pd

#预定义参数
start = "2015-1-1"
end = "2017-12-18"

#保存路径
save_path = "./pool/"
#保存示例
#df.to_excel(+ "data.xls")

#代码
stock = "600372"

#获取数据

#获取当天所有的创业板数据

try:
    today_all = ts.get_today_all()
except:
    #发生异常执行这段代码
    print ("未获得当天数据")
else:
    print ("get today_all_data...")
    today_all.to_excel(save_path + "today_all_data.xls")
    print ("save today_all_data...")


try:
    # 获取创业板的
    gems = ts.get_gem_classified()
    print ("get gem...")
except:
    #发生异常执行这段代码
    print ("未获得当天创业板数据")
else:
    gems.to_excel(save_path + "gems_data.xls")
    print ("save gems_data...")




"""
根据条件筛选合适的数据
"""

try:
    today_all_data = pd.read_excel(save_path + "today_all_data.xls")
except:
    print ("today_all_data不存在")

    today_all_data.to_excel(save_path + "select_data.xls")
else:
    try:
        gem_data = pd.read_excel(save_path + "gems_data.xls")
    except:
        print ("gems_data不存在")
    else:
        select_code = list(today_all_data.code)
        for i in today_all_data.code:
            for j in gem_data.code:
                if i == j:
                    select_code.remove(i)
        select_code.to_excel(save_path + "select_data.xls")


#选择创业板目前失效，使用当天所有的数据替代
#start_stocks = pd.read_excel(save_path + "today_all_data.xls")
start_stocks = today_all_data
print (start_stocks.head())

#股价在10~20间,(start_stocks["volume"] > 1e+08)&(start_stocks["turnoverratio"] > 1)
good_price_code = start_stocks[(start_stocks["trade"] > 10) & (start_stocks["trade"] < 20) & (start_stocks["turnoverratio"] > 2) & (start_stocks["volume"] > 1e+07)]

#将符合条件的股票数据存入excel
good_price_code.to_excel(save_path + "good_price_code.xls")

print ("The numble of good_price_code is {0}".format(len(good_price_code)))



