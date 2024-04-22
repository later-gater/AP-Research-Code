import pandas as pd
import dill

def get_df():
    df = pd.read_pickle("permutations.pkl")
    df["robot_growth_func"] = df["robot_growth_func"].apply(lambda x: dill.loads(x))
    return df

def access_rows(row_range: slice) -> pd.DataFrame:
    df = get_df()
    rows = df.iloc[row_range]
    return rows
