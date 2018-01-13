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
from quantist.data.get_and_save_data import GetDataFromTushare as gdft
from quantist.utils import change_data_form as cdf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd
import tushare as ts
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
from matplotlib.finance import candlestick_ohlc


class Plotter(object):
    """
    Plot basic parameter
    parameter: close, p_change
    Example:
        path = "../data/pool/"
        Plotter.plot_single_stock(path, 'close','600848')
    """
    def plot_single_stock(path, parameter, code=None,):
        data_gdft = gdft.getSavedStock(path, code)
        # Formed the data
        data = cdf.ChangeDataForm.to_day_form(data_gdft)
        data[parameter].plot(label=code)
        plt.legend(loc='best')
        plt.title(code)
        plt.show()

    def plot_heatmap(self,data):
        sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.1, vmin=0)
        plt.show()



    def echarts_bar(self):
        # After run we can get render.html
        from pyecharts import Bar
        attr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        v1 = [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.2, 32.6, 20.0, 6.4, 3.3]
        v2 = [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3]
        bar = Bar("Bar chart", "precipitation and evaporation one year")
        bar.add("precipitation", attr, v1, mark_line=["average"], mark_point=["max", "min"])
        bar.add("evaporation", attr, v2, mark_line=["average"], mark_point=["max", "min"])
        bar.render()

    def plot_kline(data, show_num=60, **kargs):
        """
        #Plot K line

        绘制一目平衡表指标图
        参数:
            股票代码: code
            开始时间: start
            截至时间: end
            绘制条数: show_num
            绘图周期: freq
        输出:
            时间，开，高，低，收：t, o, h, l, c
            短轴快线（转换线）: tenkan
            中轴慢线（基准线）: kijun
            后移指标（迟行带）: chinkou
            前移指标A（先行带A）: senkoua
            前移指标B（先行带B）: senkoub
        Example:
            data_path = "../data/pool/"
            result_path = "../data/result/"
            result_name = 'plot_kline'
            data = pd.read_excel(data_path + 'sh2.xls')
            plot_kline(data,show_num=120)    # sh

        :param show_num:
        :param kargs:
        :return:
        """
        def ICHIMOKU(high, low, close, shortperiod=7, mediumperiod=22, longperiod=44):
            '''
            一目平衡表指标(Ichimoku Kinko Hyo)
            输入:
                最高价序列：high
                最低价序列：low
                收盘价序列：close
            参数: 原始的默认参数组合：9， 26， 52
                短线周期: shortperiod
                中线周期: mediumperiod
                长线周期: longperiod
            输出:
                短轴快线（转换线）: tenkan
                中轴慢线（基准线）: kijun
                后移指标（迟行线）: chinkou
                前移指标A（先行带A）: senkoua
                前移指标B（先行带B）: senkoub
            '''

            def windowfun(array, fun, n):
                '''
                滑动窗口函数
                输入:
                    标的数组：array
                参数:
                    窗口函数：fun  要求可以以数组为单参数函数
                    窗口长度：n    要求 n <= len(array)
                '''

                m = len(array)
                idx = np.arange(m)
                if m < n:
                    return []

                def _fun(i):
                    if i < n - 1:
                        return np.nan
                    return fun(array[i - n + 1:i + 1])

                _fun = np.frompyfunc(_fun, 1, 1)
                return _fun(idx)

            def arrayshift(array, _n, ext=False):
                '''
                相对下标平移数组数据
                    数据移动后有下标空位，则用np.nan填充
                    移出原始下标范围之外的数据，则根据ext控制是否作为第二参数返回

                输入:
                    array：标的数组
                参数:
                    n：相对下标移动的长度（向右移为正，向左移为负）
                    ext：是否返回移出原始下标范围之外的数据，默认取False，即不返回

                '''

                new_array = array
                n = _n
                if n == 0:
                    return new_array

                if abs(n) > len(array):
                    n = np.sign(n) * len(array)

                nan_array = np.ones(abs(n)) * np.nan
                ext_array = None
                if n > 0:
                    new_array, ext_array = np.split(array, [-n])
                    new_array = np.concatenate((nan_array, new_array))
                else:
                    ext_array, new_array = np.split(array, [-n])
                    new_array = np.concatenate((new_array, nan_array))

                if ext:
                    return new_array, ext_array
                return new_array

            # 短轴快线 = 转换线
            tenkan = (windowfun(high, max, shortperiod) + windowfun(low, min, shortperiod)) / 2
            # 中轴慢线 = 基准线
            kijun = (windowfun(high, max, mediumperiod) + windowfun(low, min, mediumperiod)) / 2
            # 后移指标=迟行线
            chikou = arrayshift(close, - mediumperiod)
            # 先行带A
            senkoua = (tenkan + kijun) / 2
            senkoua, senkoua_ext = arrayshift(senkoua, mediumperiod, True)
            # 先行带B
            senkoub = (windowfun(high, np.max, longperiod) + windowfun(low, np.min, longperiod)) / 2
            senkoub, senkoub_ext = arrayshift(senkoub, mediumperiod, True)

            return tenkan, kijun, chikou, senkoua, senkoub, senkoua_ext, senkoub_ext

        # df = data

        # df = df[df.paused == False]
        t = data.index.values
        o = data['open'].values
        h = data['high'].values
        l = data['low'].values
        c = data['close'].values
        v = data['volume'].values
        tenkan, kijun, chinkou, senkoua, senkoub, senkoua_ext, senkoub_ext = ICHIMOKU(h, l, c)

        # _show_num = show_num
        _show_num = show_num
        if _show_num > len(t):
            _show_num = len(t)

        zeros = np.zeros_like(senkoua_ext)
        nans = np.ones_like(senkoua_ext) * np.nan
        # 转换成dt.datetime对应时间戳的数据序列
        _t = t[-_show_num:].astype(dt.date) / 1000000000
        _t = np.concatenate((_t, zeros))
        _o = np.concatenate((o[-_show_num:], nans))
        _h = np.concatenate((h[-_show_num:], nans))
        _l = np.concatenate((l[-_show_num:], nans))
        _c = np.concatenate((c[-_show_num:], nans))
        _v = np.concatenate((v[-_show_num:], zeros))
        _tenkan = np.concatenate((tenkan[-_show_num:], nans))
        _kijun = np.concatenate((kijun[-_show_num:], nans))
        _chinkou = np.concatenate((chinkou[-_show_num:], nans))
        _senkoua = np.concatenate((senkoua[-_show_num:], senkoua_ext))
        _senkoub = np.concatenate((senkoub[-_show_num:], senkoub_ext))

        def draw_line(t, o, h, l, c, v, *args, **kargs):
            '''
            绘制K线图，并叠加额外曲线于其上
            输入:
                时间序列：t， 要求：dt.datetime对应的时间戳序列
                开盘价序列：o
                最高价序列：h
                最低价序列：l
                收盘价序列：c
                成交量序列：v
                额外数据序列：args【可选】， 可以有多组数据序列，比如：有5组数据序列
            参数:
                额外数据序列对应的参数：kargs【可选】，只支持如下参数：
                    names 额外数据命名。比如['aa','bb','cc','dd','ee']表示额外的5条曲线对应的名字
                    colors 额外数据绘制颜色。 比如['r','b','g','r','b']表示额外的5条曲线对应的颜色
                    widths 额外数据绘制线宽。 比如['2','2','2','1','1']表示额外的5条曲线对应的线宽
                    fill_pairs 曲线间填充颜色。比如[(3,4,'y')]表示第3和4两条曲线间填充黄色
            '''

            # 准备K线数据
            t_i = np.arange(len(t))
            quotes = zip(t_i, o, h, l, c)

            # 绘制K线
            fig = plt.figure(figsize=(20, 10))
            ax1 = plt.subplot2grid((10, 4), (0, 0), rowspan=5, colspan=4)
            candlestick_ohlc(ax1, quotes, width=0.5, colorup='r', colordown='g')

            if len(args) > 0:
                # 绘制额外曲线
                names = kargs.get("names", None)
                colors = kargs.get("colors", None)
                widths = kargs.get("widths", None)
                fill_pairs = kargs.get("fill_pairs", None)
                for _i in np.arange(len(args)):
                    data = args[_i]
                    name_str = ""
                    if names:
                        name_str = ", label=\"$%s$\"" % names[_i]
                    color_str = ""
                    if colors:
                        color_str = ", \"%s\"" % colors[_i]
                    width_str = ""
                    if widths:
                        width_str = ", linewidth=%s" % widths[_i]
                    # 为exec的执行定义3个局部变量
                    scope = {'_t_i': t_i, '_ax1': ax1, "_data": data}
                    stm = "ax1.plot(t_i, data%s%s%s)" % (color_str, width_str, name_str)
                    exec(stm) in scope
                if fill_pairs:
                    for _pair in fill_pairs:
                        _i = _pair[0]
                        _j = _pair[1]
                        _c = _pair[2]
                        a = args[_i].astype(np.float)
                        b = args[_j].astype(np.float)
                        ax1.fill_between(t_i, a, b, alpha=0.5, facecolor=_c, edgecolor=_c)
            ax1.legend()

            # 绘制成交量
            ax2 = plt.subplot2grid((10, 4), (5, 0), rowspan=2, colspan=4, sharex=ax1)
            barlist = ax2.bar(t_i, v)
            for i in t_i:
                if o[i] < c[i]:
                    barlist[i].set_color('r')
                else:
                    barlist[i].set_color('g')

            # 手工设置刻度显示【可以避免candlestick_ohlc用时间作为刻度出现的跳空问题】
            plt.xlim(0, len(t) - 1)

            def t_formatter(_i, pos):
                i = int(_i)
                if i > len(t) - 1:
                    i = len(t) - 1
                elif i < 0:
                    i = 0
                if t[i] == 0:
                    return ""
                d = dt.date.fromtimestamp(t[i])
                return d.strftime('$%Y-%m-%d$')

            ax1.xaxis.set_major_formatter(tk.FuncFormatter(t_formatter))

            # result_path = kargs.get('result_path')
            # result_name = kargs.get('result_name')
            # plt.savefig(result_path+result_name)
            # 显示
            plt.show()

            draw_line(_t, _o, _h, _l, _c, _v, _tenkan, _kijun, _chinkou, _senkoua, _senkoub,
                  names=['TENKAN', 'KIJUN', 'CHINKOU', 'SENKOUA', 'SENKOUB'],
                  colors=['r', 'b', 'g', 'r', 'b'],
                  widths=[2, 2, 2, 1, 1], fill_pairs=[(3, 4, 'y')])



# Sample test
# path = "../data/pool/"
# Plotter.plot_single_stock(path, 'close','600327')

# plt.show(plt.plot([x for x in range(5)]))



#Test plot_heatmap
"""



"""
p = Plotter()
p.echarts_bar()