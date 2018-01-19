# -*- coding: utf-8 -*-

import pandas as pd
import global_list as gl
import matplotlib.pyplot as plt
import seaborn as sns

class BasicAnalysis(object):

    def basic_info(data):
        data = pd.read_csv(gl.TEST_FOREX_RESULT_PATH + 'xauusdd_form.csv')[1:]
        print(data[:8])
        print("------------------------------------------------------------------")
        print(data.describe())

    def show_p_change(data):
        data = pd.read_csv(gl.TEST_FOREX_RESULT_PATH + 'xauusdd_form.csv')[1:]
        plt.plot(data['fclose'])
        plt.show()
        plt.plot(data['p_change'])
        plt.show()


# Simple test

b = BasicAnalysis()
bi = b.basic_info()
spc = b.show_p_change()