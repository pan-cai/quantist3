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
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import pylab
# import csv
# import matplotlib.cbook as cbook

def readcodes(cfile):
    keys = []
    names = []
    for line in cfile.readlines():
        name, key = line.split(',')
        keys.append(key.strip())
        names.append(name.strip())
    cfile.close()
    codes = {'key':keys, 'name':names}
    return codes

cfile = open('index.txt', 'r',encoding='utf-8')
codes = readcodes(cfile)

startDate = datetime.datetime(2005, 4, 8)
# endDate = datetime.datetime(2016, 5, 30)
endDate = datetime.datetime.today()
startDay = str(startDate.date())
endDay = str(endDate)

def convertDate(dfValue):
    idate = []
    for i in range(0, len(dfValue.date)):
        d = dfValue.date[i]
        idate.append(datetime.datetime(int(d[0:4]), int(d[5:7]), int(d[8:])))
    dfValue.drop('date', axis=1)
    dfValue['date'] = idate

def uniform2base(df, basedf):
    startDay = df.date[0]
    # print('startDay is %s.' % startDay)
    sclose = df.close[0]
    # print('close point is %s.' % sclose)
    bdate = basedf['date'][basedf['date']==startDay]
    bclose = basedf['close'][basedf['date']==startDay]
    # print('base day is %s.' % bdate)
    # print('base point is %s.' % bclose)
    ratio = bclose/sclose
    # print(ratio)
    # print('ratio is %s, and the type is %s. ' % (ratio, type(ratio)))
    ratio = float(ratio)
    df.close = df.close * ratio

def getIndexes(startDay, codes):
    keys = list(codes['key'])
    names = list(codes['name'])
    keys.insert(0, '000300')
    names.insert(0, '沪深300')

    values =[]
    for key in keys:
        df = ts.get_k_data(key,index=True, start=startDay, end=endDay)
        df = df.reset_index(drop=True)
        values.append(df)
        convertDate(values[-1])

    for key in range(1, len(keys)):
        df = values[key]
        uniform2base(df, values[0])

    infodict = {}
    infodict['key'] = keys
    infodict['name'] = names
    infodict['value'] = values
    color = ['b', 'r', 'k', 'y', 'c', 'g', 'm']
    if len(infodict['key']) > len(color):
        print('Too many stocks. Please make it less then 5.')
        return None
    infodict['color'] = color[0:len(infodict['key'])]
    indexes = pd.DataFrame(infodict)
    return indexes

def plotIndexes(indexes):
    xsize = 9
    ysize = 6
    fig = plt.figure(figsize=(xsize,ysize))
    ax = fig.add_subplot(1,1,1)

    pylab.mpl.rcParams['font.sans-serif'] = ['SimHei']
    pylab.mpl.rcParams['axes.unicode_minus'] = False
    ax.set_title('Index Compare | DongTalks')

    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)

    datemin = startDate.date()
    datemax = endDate.date()
    ax.set_xlim(datemin, datemax)
    ax.grid(True)

    for i in range(0, len(indexes.key)):
        ax.plot(indexes.value[i]['date'], indexes.value[i]['close'], indexes.color[i], label=indexes.name[i])

    plt.tight_layout()
    ax.legend(loc=2)
    name = indexes.key[0] + 'vs' + indexes.key[1] + 'vs' + indexes.key[2] + '.png'
    plt.savefig(name,format='png')
    # plt.show()
    plt.close()

values = list(codes.values())
length = len(values[0])
for i in range(0, length, 2):
    if i == (length-1):
        i = i - 1
    key = [values[0][i], values[0][i+1]]
    name = [values[1][i], values[1][i+1]]
    subcodes = {'key':key, 'name':name}
    indexes = getIndexes(startDay, subcodes)
    plotIndexes(indexes)


# 分析最后几天的股指。可供参考，以后也可以尝试封装。
# keys = []
# idxName = []
# idxValue = []
# for key in indexName.keys():
#     keys.append(key)
#     idxName.append(indexName[key])
#     s = indexValue[key]['close']
#     lastValue = float(s[len(s)-1])
#     idxValue.append(lastValue)
# df = pd.DataFrame({'key':keys, 'name':idxName, 'value':idxValue})
# df = df.sort_values('value', ascending=False)
# # print(df)
