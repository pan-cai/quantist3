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
# 2 Construct dataform
# Example:
#     column:
#         date open close p_change ... change_base close_front capital label retracement
#     basic calculate:
#         p_change:
#             (today_close - yesterday_close)*100/today_close
#         change_base:
#             (close_today - close_base)*100/close_today
#             sh["basic_change"] = (sh["close"] - close_base_day)*100/sh["close"]
#     basic_day data:
#         date,       open,    high,   close,      low,      volume,    price_change, p_change
#         2014-12-10, 2855.94, 2946.707, 2940.006, 2807.678, 5128869.0, 83.737,        2.93
#         drawdown=max（capital_base - capital）/capital_base
#         retracement = （capital_base - capital）/capital_base
#         label
#             buy:1
#             sell:-1
#             no operate :0
#         change_rate:(open_today - close_front)/open_today
#         close_front:
#             close price of deal day
#             first close_front is  2940.006
#             close_front is natative
#     captival
#         capital_base = 10,000,000
#         account
#         account balance = capital_base
#     strategy:
#         rate of return:close price
#         buying or sell:open price
#         end trade:
#             capital < 0
#         buy--buy_rate(more than 3%)
#         sell--sell_rate(less than -3%)
#         first deal: 20% capital_base = 20%*open_base_day
#         change_rate:(open_today - close_front)/open_today
#         if change_rate < -3%:
#             buy--- 1
#             capital = captical - open_today*100
#             close_front = close_today
#         elif change_rate > 3%:
#             sell---- -1
#              capital = captical + open_today*100
#              close_front = close_today
#         else:
#             no_handle 0
#             capital = captical
#             close_front = close_front
#
#
#     other:


"""

"""
# basic date
#
# capital_base（起始资金）
# commission（手续费）
#     buycost
#     sellcost
# slippage（滑点）
#     买入股票的交易价格调整为 S × (1 + value)
#     卖出股票的交易价格调整为 S × (1 - value)
#  order(s, 100)
#
# Capitcal & balance
# share(1)---straddle(100)
# max_history_window（回溯长度）

"""
"""
# change_base:
#     (sh["close"] - close_base)/sh["close"]
#     Calculate the maximum retracement
# close_font: closed price of latest trade
# change_rate:rate between now and close_font
# mark:
#     change_rate > sell_price, mark=1, sell
#     change_rate < buy_price, mark=-1, buy
#     others,                  mark=0, no handler
#
# Example data form
#         change_base  change_rate     close  close_font       date  mark      open  \
#     0      0.000000    -0.029435  2940.006    2940.006 2014-12-10    -1  2855.940
#     1     -0.487500    -0.009497  2925.743    2925.743 2014-12-11    -1  2912.346
"""
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

"""
# Get hs300 data
sql = "select * from hs300"
pd.read_sql_query(sql, engine).to_excel(data_path + "hs300.xls")
ts.get_hist_data("hs300").to_excel(data_path + "hs3002.xls")
"""

print("----------------------Mark the data------------------------------------------------------------------")

mark_buy = 1
mark_sell = -1
mark_no_hanler = 0
close_buy = []
close_sell = []

close_base = sh.ix[0, "close"]
sh["change_base"] = 0
# print("change_base  ", "  mark", "p_change")

"""
# Calculate mark from change_base, change_base is template paramerter
# Mark is based on close
"""
"""
for row in range((len(sh["close"]))):
    sh.ix[row,"change_base"] = (sh.ix[row,"close"] - close_base) / sh.ix[row, "close"]
    if sh.ix[row, "change_base"] > 0.03:
        close_base = sh.ix[row, "close"]
        sh.ix[row, "mark"] = mark_sell
        close_sell.append(row)
        close_sell.append(close_base)
        #print(close_sell)
    elif sh.ix[row, "change_base"] < -0.03:
        close_base = sh.ix[row, "close"]
        sh.ix[row, "mark"] = mark_buy
        close_buy.append(row)
        close_buy.append(close_base)
    else:
        sh.ix[row, "mark"] = mark_no_hanler
"""

# print(sh.ix[row, "change_base"], sh.ix[row, "mark"], sh.ix[row, "p_change"])

"""
# Mark is based on open
"""
for row in range((len(sh["close"]))):
    sh.ix[row, "change_base"] = (sh.ix[row, "open"] - close_base) / sh.ix[row, "open"]
    if sh.ix[row, "change_base"] > 0.03:
        close_base = sh.ix[row, "open"]
        sh.ix[row, "mark"] = mark_sell
        close_sell.append(row)
        close_sell.append(close_base)
        # print(close_sell)
    elif sh.ix[row, "change_base"] < -0.03:
        close_base = sh.ix[row, "open"]
        sh.ix[row, "mark"] = mark_buy
        close_buy.append(row)
        close_buy.append(close_base)
    else:
        sh.ix[row, "mark"] = mark_no_hanler
# print(result[:100])

# print(close_buy)

plt.plot(sh["close"], label="close")
close_sell_x = [x for x in close_sell if close_sell.index(x) % 2 == 0]
close_sell_y = [y for y in close_sell if close_sell.index(y) % 2 != 0]
close_buy_x = [x for x in close_buy if close_buy.index(x) % 2 == 0]
close_buy_y = [y for y in close_buy if close_buy.index(y) % 2 != 0]

"""
print(len(close_buy))
print(close_buy_x)
print(close_buy_y)
"""

plt.scatter(close_sell_x, close_sell_y, label="close_sell")
plt.scatter(close_buy_x, close_buy_y, label="close_buy")
plt.legend(loc="best")
plt.savefig(result_path + "strategy_grid.jpg")
plt.show()

print("----------------------Get trade and balance----------------------------------------------------------")

date_base = 2014 - 12 - 10
close_base = sh.ix[0, "close"]  # 2940.006
capital = 10000000
# sh["balance"] = 10000000
sell_price = 0.3  # If change_rate > sell_rate, sell the stock
buy_price = -0.3  # If change_rate > sell_rate, sell the stock
sell_rate = 0.1  # count/balance???
buy_rate = 0.1  # count/balance???

buy_cost = 0.0003
sell_commission = 0.0013
min_cost = 5

buy_first_arte = 0.2  # Buy rate at first trade

"""
# Calculate balance

# open和前天的close比较得到mark
# mark=0,无操作，amoun=0,balance=前天balance
# mark=1,买入，amoun=100,balance=(前天balance - open*amount)
# mark=-1,卖出，amoun=0,balance=(前天balance + open*amount)
"""
# initializtion
balance = 0
base_amount = 100
close_list = [c for c in range((len(sh["close"])))]
"""
# The first day and end day
"""
sh.ix[0, "trade_amount"] = capital * buy_first_arte / sh.ix[0, "open"]
sh.ix[0, "balance"] = capital - sh.ix[0, "trade_amount"] * sh.ix[0, "open"]
# print(sh.ix[row, "trade_amount"], sh.ix[row, "balance"])
balance = sh.ix[0, "balance"]

for row in close_list[1:len(sh["close"])]:  # row 1~731, total num is 733 (0~732)
    # print(row)
    if sh.ix[row, "mark"] == mark_no_hanler:
        sh.ix[row, "trade_amount"] = 0
        sh.ix[row, "balance"] = sh.ix[(row - 1), "balance"]
        balance = sh.ix[(row - 1), "balance"]
    elif sh.ix[row, "mark"] == mark_buy:
        sh.ix[row, "trade_amount"] = base_amount
        sh.ix[row, "balance"] = balance - sh.ix[row, "trade_amount"] * sh.ix[row, "open"]
        balance = sh.ix[row, "balance"]
    elif sh.ix[row, "mark"] == mark_sell:
        sh.ix[row, "trade_amount"] = base_amount
        sh.ix[row, "balance"] = balance + sh.ix[row, "trade_amount"] * sh.ix[row, "open"]
        balance = sh.ix[row, "balance"]
    else:
        print("wrong-------")

result = {"date": sh["date"], "open": sh["open"], "close": sh["close"],
          "change_base": sh["change_base"], "mark": sh["mark"], "balance": sh["balance"],
          "trade_amount": sh["trade_amount"]}
result = pd.DataFrame(result)

"""
# Save the result
"""

result.to_csv(result_path + "strategy_grid.csv")
print("Strategy grid have saved to " + result_path + "strategy_grid.csv")

print("----------------------Analysis the result------------------------------------------------------------------")

"""
# Analysis the result
# returns: 列表。策略日收益率。
# cumulative_returns: 列表。策略累计收益率。
# cumulative_values: 列表。策略累计价值。
# annualized_return: 浮点。策略年化收益率。
# excess_return: 浮点。策略相对无风险收益率的超额收益。
# volatility: 浮点。策略年化波动率。
# max_drawdown: 浮点。策略最大回撤。
# benchmark_returns: 列表。参考标准日收益率。
# benchmark_cumulative_returns: 列表。参考标准累计收益率。
# benchmark_cumulative_values: 列表。参考标准累计价值（初始值为1）。
# benchmark_annualized_return：参考标准年化收益率。
# treasury_return: 同期无风险收益率。
# alpha: 策略CAPM阿尔法。
# beta: 策略CAPM贝塔。
# sharpe: 策略年化夏普率。
# information_coefficient: 信息系数。
# information_ratio: 信息比率。
# turnover_rate: 换手率。

# {'alpha': -0.27802826887619814,
#  'annualized_return': -0.3304053715840608,
#  'benchmark_annualized_return': -0.94569692710343789,
#  'benchmark_cumulative_returns': tradeDate
#  2014-01-06   -0.022761
#  2014-01-07   -0.023036
#  dtype: float64,
#  'benchmark_cumulative_values': tradeDate
#  2014-01-06    0.977239
#  2014-01-07    0.976964
#  dtype: float64,
#  'benchmark_returns': tradeDate
#  2014-01-06   -0.022761
#  2014-01-07   -0.000281
#  dtype: float64,
#  'benchmark_volatility': 0.25132826716445777,
#  'beta': 0.099752707060321244,
#  'cumulative_returns': tradeDate
#  2014-01-06   -0.002724
#  2014-01-07   -0.003204
#  dtype: float64,
#  'cumulative_values': tradeDate
#  2014-01-06    0.997276
#  2014-01-07    0.996796
#  dtype: float64,
#  'excess_return': -0.37701337158406079,
#  'information_coefficient': 5.479787715404199,
#  'information_ratio': 10.959575430808398,
#  'max_drawdown': 0.0032035200000000152,
#  'returns': tradeDate
#  2014-01-06   -0.002724
#  2014-01-07   -0.000481
#  dtype: float64,
#  'sharpe': -15.038022367852053,
#  'treasury_return': 0.046608000000000004,
#  'turnover_rate': 0.0,
#  'volatility': 0.025070675010434321}


年化收益率是指投资期限为一年所获的收益率。
年化收益率=（投资内收益/本金）/（投资天数/365）×100%
年化收益=本金×年化收益率
实际收益=本金×年化收益率×投资天数/365
Annualized yield = (return on investment / principal) / (investment days / 365) × 100%
Annualized income = principal × annualized rate of return
Real income = principal × annualized yield × investment days / 365
"""

"""
grid_data = pd.read_csv(result_path + "strategy_grid.csv")
#print(grid_data[:3])
# plt.plot(grid_data["close"])
# plt.show()
plt.plot(grid_data["balance"])
plt.show()

# treasury_return: 同期无风险收益率(Risk-free rate of return)。
# annualized_return: 浮点。策略年化收益率。

#hs300
hs300 = pd.read_excel(data_path + "hs3002.xls")[0:(len(grid_data)-1)] # The length is different
# print(len(hs300))
# print(hs300[:3])

# annualized_return
annualized_return = np.power((grid_data.ix[0,"balance"]/grid_data.ix[len((grid_data["balance"])-1),"balance"]),250.0/1068) -1
benchmark_annualized_return = np.power((hs300.ix[0,"close"]/hs300.ix[len((hs300["close"])-1),"close"]),250.0/1068) -1
treasury_return = 0.4
beta = np.cov(sh["p_change"],hs300["p_change"])/hs300["p_change"].var()
alpha = annualized_return - treasury_return - beta*(benchmark_annualized_return - treasury_return)
#Volatility = np.power((250.0*..../1068), 0.5)
# sharpe = (annualized_return - treasury_return)/volatility
# information_ratio = (annualized_return - benchmark_annualized_return)/....
# max_drawdown = np.max(1 - )
# turnover_rate = ...
"""
