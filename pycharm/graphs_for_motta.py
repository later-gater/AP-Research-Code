import pandas as pd
from numpy import isclose
import matplotlib.pyplot as plt

df = pd.read_pickle("data_1.pkl")


def graph(chosen_wage, chosen_rate):

    filters = [df["L"] > 100, pd.Series(isclose(df["w"], chosen_wage)), pd.Series(isclose(df["r"], chosen_rate))]  # floating point error, change to np.is_near

    filtered_df = df
    for filter in filters:
        filtered_df = filtered_df[filter]

    fig = plt.figure()

    fig.suptitle(f"w: {chosen_wage}, r: {chosen_rate}")

    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("Demands")
    ax.plot(filtered_df["p"], filtered_df["L"], color="red", label="Labor")
    ax.plot(filtered_df["p"], filtered_df["K"], color="blue", label="Capital")
    ax.legend(loc="upper center")
    ax.set_xlabel("Price")
    ax.set_ylabel("Quantity Demanded")

    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("Costs")
    ax.plot(filtered_df["p"], filtered_df["LC"], color="red", label="Labor Costs")
    ax.plot(filtered_df["p"], filtered_df["KC"], color="blue", label="Capital Costs")
    ax.plot(filtered_df["p"], filtered_df["TC"], color="black", label="Total Costs")
    ax.legend(loc="upper center")
    ax.set_xlabel("Price")
    ax.set_ylabel("Cost of Inputs")

    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("Profits")
    ax.plot(filtered_df["p"], filtered_df["R"], color="red", label="Total Revenue")
    ax.plot(filtered_df["p"], filtered_df["TC"], color="blue", label="Total Costs")
    ax.plot(filtered_df["p"], filtered_df["P"], color="black", label="Profits")
    ax.legend(loc="upper center")
    ax.set_xlabel("Price")
    ax.set_ylabel("$")


    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("Average Costs")  # this is incorrect :( TC/R instead of TC/Y
    ax.plot(filtered_df["p"], filtered_df["AC"], color="red", label="Average Costs")
    ax.legend(loc="upper center")
    ax.set_xlabel("Price")
    ax.set_ylabel("Average Costs")

    plt.show()




pass
