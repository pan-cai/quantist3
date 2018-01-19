# -*- coding: utf-8 -*-

from unittest import TestCase
import pandas as pd
import global_list as gl
from utils import preproce_data

class PreproceDataTest(TestCase):

    def test_forex_to_standard(self):

        xauusdd = pd.read_csv(gl.TEST_FOREX_XAUUSDD_DATA)
        save_name = 'xauusdd_form.csv'
        result_data_path = gl.TEST_FOREX_RESULT_PATH
        preproce_data.forex_to_standard(xauusdd, save_name, result_data_path, saved=True)