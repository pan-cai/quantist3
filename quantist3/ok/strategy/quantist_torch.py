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
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
import numpy as np


"""
Basic example
"""
class QuantistTorch(object):

    def demo1(self):
        data_path = "../data/pool/"
        result_path = "../data/result/"
        data = pd.read_excel(data_path + "Google_Stock_Price_Test.xls")

        x = np.array(data['close'])
        y = np.array(data['open'])

        X = Variable(torch.from_numpy(x)).float()
        y = Variable(torch.from_numpy(y)).float()

        W = Variable(torch.Tensor([-1]), requires_grad=True)
        b = Variable(torch.Tensor([-2]), requires_grad=True)

        plt.scatter(X.data.numpy(), y.data.numpy())
        predictions = W * X + b
        plt.plot(X.data.numpy(), predictions.data.numpy(), color='g', label='initial model')
        plt.title('Toy Data + Noise')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.show()

        import torch.nn as nn
        import torch.optim as optim

        error_fxn = nn.MSELoss()
        learning_rate = 0.01
        optimizer = optim.SGD([W, b], lr=learning_rate)

        losses = []

        num_iterations = 1000
        for iteration in range(num_iterations):
            predictions = W * X + b
            loss = error_fxn(predictions, y)
            losses.append(loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        plt.plot(losses)
        plt.title('Training plot')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        plt.grid()
        plt.show()

        plt.scatter(X.data.numpy(), y.data.numpy())
        predictions = W * X + b
        plt.plot(X.data.numpy(), predictions.data.numpy(), color='g', label='final model')
        plt.title('Toy Data + Noise')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.show()

