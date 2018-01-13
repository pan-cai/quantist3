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

from unittest import TestCase
from quantist.strategy import quantist_keras
import pandas as pd
from quantist.strategy.quantist_keras import QuantistKeras

class TestQuantistKeras(TestCase):



    def test_demo_google(self):
        data_path = "../quantist/data/pool/"
        training_set = pd.read_excel(data_path + 'Google_Stock_Price_Train.xls')
        test_set = pd.read_excel(data_path + 'Google_Stock_Price_Test.xls')
        q2 = QuantistKeras()
        q2.demo_google(training_set, test_set)