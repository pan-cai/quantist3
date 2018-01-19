import pandas  as pd

data_path = "../data/pool/"

sh = pd.read_csv(data_path + "sh.csv").sort_index(ascending=False)

print(sh)