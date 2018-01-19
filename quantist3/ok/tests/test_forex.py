# -*- coding: utf-8 -*-
import global_list as gl
import pandas as pd
import matplotlib.pyplot as plt

data_path = gl.TEST_FOREX_DATA_PATH

print(data_path)

data = pd.read_csv(data_path + 'data/XAUUSD-d.csv')
print(data[:3])

plt.plot(data['fclose'])
plt.show()
