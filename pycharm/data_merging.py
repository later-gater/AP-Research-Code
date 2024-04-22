import os
import pandas as pd
import numpy as np
import re

data = pd.DataFrame(columns= ['time', 'class', 'Convergence Error', 'Money in Circulation', 'Firm Money',
                              'Leftover Goods', 'Firm Labor Demand', 'Firm Capital Demand', 'Firm Robot Demand',
                              'Robot Growth', 'Price', 'Wage', 'Consumer Start Moneys', 'Utils', 'Times Worked',
                              'Leisure', 'Money Earned', 'Goods Purchased', 'Money Spent', 'Money Saved',
                              'Interest Earned', 'alpha2', 'eos1', 'eos2', 'robot_growth', 'income_inequality',
                              'rate_k', 'rate_z'])

perms = pd.read_pickle("permutations.pkl")

files = os.listdir('data')
files.sort(key=lambda x: int(re.findall(r'\d+_', x)[0][:-1]))

for i, filename in enumerate(files):
    df = pd.read_pickle(f"data/{filename}")

    index = perms.iloc[i]["file_name"].split("_")
    index.insert(0, i)
    names = ["index", "alpha2", "eos1", "eos2", "robot_growth", "income_inequality", "rate_k", "rate_z"]
    for j in range(len(names)):
        df[names[j]] = index[j]
    df["time"] = df.index.map(lambda x: x[0])
    df["class"] = df.index.map(lambda x: x[1])
    df["index"] = df["index"].astype(int)
    df = df.reset_index(drop=True)
    df.set_index(["index"], inplace=True)
    data = pd.concat([data, df])

data.sort_index(inplace=True, kind="stable")
data.to_pickle("raw_data.pkl")

filtered_data = data[(np.isclose(data["Firm Labor Demand"], 0.2) == False)
                     & (np.isclose(data["Firm Robot Demand"], 0.2) == False)
                     & (np.isclose(data["Firm Capital Demand"], 0.2) == False)
                     & (data["Convergence Error"] < 0.1)]

filtered_data.to_pickle("filtered_data.pkl")