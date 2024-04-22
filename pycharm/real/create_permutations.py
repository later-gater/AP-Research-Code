import pandas as pd
import itertools
import dill

alpha2_perm = {"low": 0.1, "mid": 0.5, "high": 0.9}
eos1_perm = {"low": 0.1, "lower": 0.4, "higher": 1.2, "high": 1.8}
eos2_perm = {"low": 0.4, "lower": 0.8, "higher": 3.6, "high": 10.}
robot_growth_func_perm = {"none": dill.dumps(lambda t: 1), "linear": dill.dumps(lambda t: t), "2exp": dill.dumps(lambda t: 2**t)}
income_inequality_perm = {"high": [45, 10, 5], "mid": [30, 20, 10], "low": [25, 20, 15]}
rate_k_perm = {"low": 3, "high": 9}
rate_z_perm = {"low": 2, "high": 7}


permutations = itertools.product(alpha2_perm.items(), eos1_perm.items(), eos2_perm.items(),
                                 robot_growth_func_perm.items(), income_inequality_perm.items(), rate_k_perm.items(),
                                 rate_z_perm.items())

df = pd.DataFrame(list(permutations), columns=["alpha_2", "eos_1", "eos_2", "robot_growth_func", "incomes", "rate_k", "rate_z"])

file_name_column = df.apply(lambda row: "_".join([cell[0] for cell in row]), axis=1)
df = df.applymap(lambda cell: cell[1])
df["file_name"] = file_name_column


df.to_pickle("permutations.pkl")

