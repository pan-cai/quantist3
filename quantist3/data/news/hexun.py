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

import urllib.request
import json

liveNewsUrl=r'http://newsapi.eastmoney.com/kuaixun/v1/getlist_102_ajaxResult_50_1_.html?r=0.29795602867346616'

liveNewsData=urllib.request.urlopen(liveNewsUrl)
news=liveNewsData.read()
news=news[15:].decode()

a=json.loads(news)
b=a['LivesList']

print(b)