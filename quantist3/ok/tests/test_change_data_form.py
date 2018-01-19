# -*- coding: utf-8 -*-
from unittest import TestCase
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd
import global_list as gl

class TestChangeDataForm(TestCase):

    def test_forex_basic(self):

        data = gl.TEST_FOREX_XAUUSDD_DATA
        data = pd.read_csv(data)

        print(data[:3])

