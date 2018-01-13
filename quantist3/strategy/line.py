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
Description:Line-Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels import regression




class LinearRegression(object):
    def ols_example(self):
        nobs = 20
        X = np.random.random((nobs, 2))
        X = sm.add_constant(X)
        beta = [1, 1., .5]
        #e = np.random.random(nobs)
        # y = np.dot(X, beta)
        y = np.dot(X, beta)
        # e = np.random.random(nobs)
        print("-------x------------------")
        print(X)
        print("-------y------------------")
        print(y)

        results = sm.OLS(y, X).fit()
        print(results.summary())

    def sm_ols(data):
        X = [x for x in range(len(data))]
        y = np.array(data)
        model = sm.OLS(y, X)
        # 这里做回归就是直接线性回归，因为行业之间有相关性，可以先验证多重共线性的严重程度，
        # 再利用岭回归和主成分分析消缓多重共线性
        results = model.fit()
        print(results.params)
        print(results.summary())
        y_fitted = results.fittedvalues
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(X, y, 'o', label='data')
        ax.plot(X, y_fitted, 'r--.', label='OLS')
        ax.legend(loc='best')
        plt.show()

    def linreg(X, Y):
        # Running the linear regression
        X = sm.add_constant(X)
        model = regression.linear_model.OLS(Y, X).fit()
        a = model.params[0]
        b = model.params[1]
        X = X[:, 1]
        # Return summary of the regression and plot results
        X2 = np.linspace(X.min(), X.max(), 100)
        Y_hat = X2 * b + a
        plt.scatter(X, Y, alpha=0.3)  # Plot the raw data
        plt.plot(X2, Y_hat, 'r', alpha=0.9);  # Add the regression line, colored in red
        plt.xlabel('X Value')
        plt.ylabel('Y Value')
        return model.summary()


"""
Simple test
"""
data_path = "../data/pool/"
result_path = "../data/result/"

# Test ols_example
# t = LinearRegression()
# t.ols_example()
# print(t)

# Test sm_ols
data = pd.read_excel(data_path + "Google_Stock_Price_Test.xls")
print(list(data['close'])[0:3])
LinearRegression.sm_ols(data['close'])

"""
# Test 
# asset = get_price('601857.XSHG', start_date=start, end_date=end, fields=['close'])
# benchmark = get_price('000001.XSHE', start_date=start, end_date=end, fields=['close'])
# # We have to take the percent changes to get to returns
# # Get rid of the first (0th) element because it is NAN
# r_a = asset.pct_change()[1:]
# r_b = benchmark.pct_change()[1:]
# linreg(r_b.values, r_a.values)
# 
# X = np.random.rand(100)
# Y = np.random.rand(100)
# linreg(X, Y)
# 
# # Generate ys correlated with xs by adding normally-destributed errors
# Y = X + 0.2*np.random.randn(100)
# linreg(X,Y)
# 
# import seaborn
# start = '2014-10-01'
# end = '2015-10-01'
# asset = get_price('601857.XSHG', start_date=start, end_date=end, fields=['close'])
# benchmark = get_price('000001.XSHE', start_date=start, end_date=end, fields=['close'])
# # We have to take the percent changes to get to returns
# # Get rid of the first (0th) element because it is NAN
# r_a = asset.pct_change()[1:]
# r_b = benchmark.pct_change()[1:]
# # Flatten r_a.values and r_b.values because joinquant returns nested lists
# # which is not suitable for seaborn plotting
# # Added by chihungfei
# flatten_a = np.array([item for sublist in r_a.values for item in sublist])
# flatten_b = np.array([item for sublist in r_b.values for item in sublist])
# seaborn.regplot(flatten_b, flatten_a);
"""
