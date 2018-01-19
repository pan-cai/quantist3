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
Description:Sea turtle trading
When the price more than 20 day moving average, buy
When the price more than 10 day moving average, sell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
from sqlalchemy import create_engine
engine = create_engine('mysql+pymysql://root:caicai520@127.0.0.1/quantist?charset=utf8')

print("----------------------Get the data--------------------------------------------------------------------")
"""
 1 Get data from pool
"""

data_path = "../data/pool/"
result_path = "../data/result/"

sh = pd.read_csv(data_path + "sh.csv").sort_index(ascending=False)
sh.reset_index(inplace=True)
sh['date'] = pd.to_datetime(sh['date'])

print("The length of data is " + str(len(sh["close"])))

mark_buy = 1
mark_sell = -1
mark_no_hanler = 0
close_buy = []
close_sell = []

close_base = sh.ix[0, "close"]
sh["change_base"] = 0

"""
Calculate ma20 and ma10
"""

# sh["ma10"] = pd.rolling_mean(sh["close"],10)
# sh["ma20"] = pd.rolling_mean(sh["close"],20)

sh["ma10"] = sh["close"].rolling(center=False,window=10).mean()
sh["ma20"] = sh["close"].rolling(center=False,window=20).mean()

sh["diff10"] = sh["open"] - sh["ma10"]
sh["diff20"] = sh["open"] - sh["ma20"]

#print(sh[:30])


"""
# Mark is based on open
"""
num = len(sh["close"])
for row in range(num)[19:num]:
    if sh.ix[row, "diff20"] > 0:
        close_base = sh.ix[row, "open"]
        sh.ix[row, "mark20"] = mark_buy
        close_buy.append(row)
        close_buy.append(close_base)
    else:
        sh.ix[row, "mark20"] = mark_no_hanler

    if sh.ix[row, "diff10"] < 0:
        close_base = sh.ix[row, "open"]
        sh.ix[row, "mark10"] = mark_sell
        close_sell.append(row)
        close_sell.append(close_base)
    else:
        sh.ix[row, "mark10"] = mark_no_hanler

plt.plot(sh["close"], label="close")
plt.plot(sh["ma10"], label="ma10")
plt.plot(sh["ma20"], label="ma20")
close_sell_x = [x for x in close_sell if close_sell.index(x) % 2 == 0]
close_sell_y = [y for y in close_sell if close_sell.index(y) % 2 != 0]
close_buy_x = [x for x in close_buy if close_buy.index(x) % 2 == 0]
close_buy_y = [y for y in close_buy if close_buy.index(y) % 2 != 0]

plt.scatter(close_sell_x, close_sell_y, label="close_sell")
plt.scatter(close_buy_x, close_buy_y, label="close_buy")
plt.legend(loc="best")
plt.savefig(result_path + "strategy_turtle.jpg")
plt.show()

print("----------------------Get trade and balance----------------------------------------------------------")
print("From the strategy_turtle picture we can see that this is not a good idea.")


# date_base = 2014 - 12 - 10
# close_base = sh.ix[0, "close"]  # 2940.006
# capital = 10000000
# # sh["balance"] = 10000000
# sell_price = 0.3  # If change_rate > sell_rate, sell the stock
# buy_price = -0.3  # If change_rate > sell_rate, sell the stock
# sell_rate = 0.1  # count/balance???
# buy_rate = 0.1  # count/balance???
#
# buy_cost = 0.0003
# sell_commission = 0.0013
# min_cost = 5
#
# buy_first_arte = 0.2  # Buy rate at first trade

"""
# Calculate balance
"""
# initializtion
# balance = 0
# base_amount = 100
# close_list = [c for c in range((len(sh["close"])))]
"""
# The first day and end day
"""
# sh.ix[0, "trade_amount"] = capital * buy_first_arte / sh.ix[0, "open"]
# sh.ix[0, "balance"] = capital - sh.ix[0, "trade_amount"] * sh.ix[0, "open"]
# # print(sh.ix[row, "trade_amount"], sh.ix[row, "balance"])
# balance = sh.ix[0, "balance"]


# for row in close_list[19:len(sh["close"])]:  # row 1~731, total num is 733 (0~732)
#     # print(row)
#     if sh.ix[row, "mark20"] == mark_no_hanler:
#         sh.ix[row, "trade_amount"] = 0
#         sh.ix[row, "balance"] = sh.ix[(row - 1), "balance"]
#         balance = sh.ix[(row - 1), "balance"]
#     elif sh.ix[row, "mark20"] == mark_buy:
#         sh.ix[row, "trade_amount"] = base_amount
#         sh.ix[row, "balance"] = balance - sh.ix[row, "trade_amount"] * sh.ix[row, "open"]
#         balance = sh.ix[row, "balance"]
#     elif sh.ix[row, "mark20"] == mark_sell:
#         sh.ix[row, "trade_amount"] = base_amount
#         sh.ix[row, "balance"] = balance - sh.ix[row, "trade_amount"] * sh.ix[row, "open"]
#         balance = sh.ix[row, "balance"]
#     else:
#         print("wrong-------")
#
#     if sh.ix[row, "mark10"] == mark_no_hanler:
#         sh.ix[row, "trade_amount"] = 0
#         sh.ix[row, "balance"] = sh.ix[(row - 1), "balance"]
#         balance = sh.ix[(row - 1), "balance"]
#     elif sh.ix[row, "mark10"] == mark_buy:
#         sh.ix[row, "trade_amount"] = base_amount
#         sh.ix[row, "balance"] = balance - sh.ix[row, "trade_amount"] * sh.ix[row, "open"]
#         balance = sh.ix[row, "balance"]
#     elif sh.ix[row, "mark10"] == mark_sell:
#         sh.ix[row, "trade_amount"] = base_amount
#         sh.ix[row, "balance"] = balance - sh.ix[row, "trade_amount"] * sh.ix[row, "open"]
#         balance = sh.ix[row, "balance"]
#     else:
#         print("wrong-------")
#
#
# result = {"date": sh["date"], "open": sh["open"], "close": sh["close"],
#           "change_base": sh["change_base"], "mark10": sh["mark10"],"mark20": sh["mark20"],
#           "balance": sh["balance"],"trade_amount": sh["trade_amount"]}
# result = pd.DataFrame(result)

"""
# Save the result
"""

# result.to_csv(result_path + "strategy_turtle.csv")
# print("Strategy turtle have saved to " + result_path + "strategy_turtle.csv")

print("----------------------Analysis the result------------------------------------------------------------------")
