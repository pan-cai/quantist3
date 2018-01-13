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
Create date: ${DATA}
"""
from unittest import TestCase
from quantist.data import get_and_save_data as gsd

"""
Description:
"""


class TestGetDataFromTushare(TestCase):
    pass

    code = "600848"
    data = gsd.GetDataFromTushare.getStockData(code)
    print(data)


class TestGetDataFromTushare(TestCase):
    def test_getStockData(self):
        self.fail()
