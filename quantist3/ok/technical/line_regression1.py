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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets,linear_model

diabetes = datasets.load_diabetes()
print(diabetes)

diabetes_x = diabetes.data[:,np.newaxis]
diabetes_x_temp = diabetes_x[:,:,2]

diabetes_x_train = diabetes_x_temp[:-20]
diabetes_x_test = diabetes_x_temp[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()

regr.fit(diabetes_x_train, diabetes_y_train)

print("Coefficients:\n",regr.coef_) # [ 938.23786125]
print("Residual sum of squares:%.2f"
      %np.mean((regr.predict(diabetes_x_test)-diabetes_y_test)**2))#Residual sum of squares:301.26
print("Variance score:%.2f"%regr.score(diabetes_x_test, diabetes_y_test))#Variance score:0.61

plt.scatter(diabetes_x_test, diabetes_y_test,color="black")
plt.plot(diabetes_x_test, regr.predict(diabetes_x_test), color="blue",linewidth=3)

plt.xticks(())
plt.yticks(())

path = "../data/result/"
plt.savefig(path + "line_regression1.jpg")
plt.show()
