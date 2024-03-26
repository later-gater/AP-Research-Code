import numpy as np
from scipy.optimize import fsolve, minimize, OptimizeResult, least_squares, brute
import matplotlib.pyplot as plt
import pandas as pd

def production_function(variables, A, alpha, time) -> int:
    labor, capital = variables
    eos = 0.4
    l_up = -eos * ((-labor / time) + np.e)
    k_up = -eos * ((-capital / time) + np.e)
    inner_term = (alpha * (labor ** l_up)) + ((1 - alpha) * (capital ** k_up))
    # print(f"labor: {labor}, capital: {capital}, l_up: {l_up}, k_up: {k_up}, inner_term: {inner_term}")
    return A * (inner_term ** (-1 / eos))

def profit_function(variables, A, alpha, time, wage, rate, price) -> int:
    labor, capital = variables
    revenue = price * production_function(variables, A, alpha, time)
    costs = wage * labor + rate * capital
    profit = revenue - costs
    # print(f"revenue: {revenue}, costs: {costs}, profit: {profit}")
    return profit

def get_max_profit(A, alpha, time, wage, rate, price) -> (OptimizeResult, int):
    minimized = minimize(lambda x: -1 * profit_function(x, A, alpha, time, wage, rate, price), np.array([10, 10]), bounds=[(0.2, None), (0.2, None)])
    return minimized, production_function(minimized.x, A, alpha, time)

def product_demand(price) -> int:
    return 10 - (0.5*price)

def labor_supply(wage) -> int:
    return -1 + wage**1.2

def equilibrium_curve_2(variables, A, alpha, time, rate):
    price, wage = variables
    max_profit, production = get_max_profit(A, alpha, time, wage, rate, price)
    quantity_curve = (production - product_demand(price)) ** 2
    labor_curve = (labor_supply(wage) - max_profit.x[0]) ** 2
    print(f"Price: {price}, Wage: {wage}, Quantity: {quantity_curve}, Labor: {labor_curve}")
    return [quantity_curve, labor_curve]

def equilibrium_curve_test(variables, A, alpha, time, rate):
    price, wage = variables
    max_profit, production = get_max_profit(A, alpha, time, wage, rate, price)
    quantity_curve = (production - product_demand(price)) ** 2
    labor_curve = (labor_supply(wage) - max_profit.x[0]) ** 2
    print(f"Price: {price}, Wage: {wage}, Quantity: {quantity_curve}, Labor: {labor_curve}, Cost: {quantity_curve + labor_curve}")
    return quantity_curve + labor_curve

def find_equilibrium_2(A, alpha, time, rate):
    return least_squares(equilibrium_curve_2, np.array([10, 10]), args=(A, alpha, time, rate), bounds=[(0.2, 0.2), (100, 100)], xtol=1e-10, ftol=1e-10, gtol=1e-10)

def equilibrium_curve(price, A, alpha, time, wage, rate):
    max_profit, production = get_max_profit(A, alpha, time, wage, rate, price)
    point =  production - product_demand(price)
    # print(f"Price: {price}, Point: {point}")
    return point

def find_equilibrium(A, alpha, time, wage, rate):
    return fsolve(equilibrium_curve, np.array([10]), args=(A, alpha, time, wage, rate))

def main():
    A = 5
    alpha = 0.75
    time = 1
    wage = 1.2
    rate = 6
    price = 2.5

    foo = find_equilibrium_2(A, alpha, time, rate)
    print("\n\n\n\n\n")
    bar = minimize(equilibrium_curve_test, np.array([10, 10]), bounds=((0.2, None), (0.2, None)), args=(A, alpha, time, rate))
    pass

    # print(production_function([10, 10], A, alpha, time))
    # L = np.linspace(0.1, 5, 50)
    # K = np.linspace(0.1, 5, 50)
    # inputs = np.meshgrid(L, K)
    #
    #
    # for price in np.linspace(1, 20, 10):
    #     print(f"Price: {price}")
    #     productions = production_function(inputs, A, alpha, time)
    #     costs = wage * inputs[0] + rate * inputs[1]
    #     profits = profit_function(inputs, A, alpha, time, wage, rate, price)
    #     max_profit = get_max_profit(A, alpha, time, wage, rate, price)
    #     print(max_profit.fun, max_profit.x)
    #     fig = plt.figure()
    #
    #     ax = fig.add_subplot(1, 3, 1, projection="3d")
    #     ax.plot_surface(inputs[0], inputs[1], productions, cmap="viridis")
    #     ax.scatter(max_profit.x[0], max_profit.x[1], -max_profit.fun, color="red")
    #     ax = fig.add_subplot(1, 3, 2, projection="3d")
    #     ax.plot_surface(inputs[0], inputs[1], costs, cmap="viridis")
    #     ax.scatter(max_profit.x[0], max_profit.x[1], -max_profit.fun, color="red")
    #     ax = fig.add_subplot(1, 3, 3, projection="3d")
    #     ax.plot_surface(inputs[0], inputs[1], profits, cmap="viridis")
    #     ax.scatter(max_profit.x[0], max_profit.x[1], -max_profit.fun, color="red")
    #     plt.show()

    # price_wage = np.meshgrid(np.linspace(1, 7.5, 100), np.linspace(1, 7.5, 100))
    # profits = np.full_like(price_wage[0], fill_value=None, dtype=np.dtype)
    # dict = []
    # for i in range(len(price_wage[0])):
    #     for j in range(len(price_wage[0][i])):
    #         minimized, production = get_max_profit(A, alpha, time, price_wage[1][i][j], rate, price_wage[0][i][j])
    #         profits[i][j] = -minimized.fun
    #         print(f"Price: {price_wage[0][i][j]}, Wage: {price_wage[1][i][j]}, Production: {production}, Profit: {minimized.fun}, Labor: {minimized.x[0]}, Capital: {minimized.x[1]}")
    #         # store what is printed in a vals dataframe
    #         dict.append({
    #             "Price": price_wage[0][i][j],
    #             "Wage": price_wage[1][i][j],
    #             "Production": production,
    #             "Profit": -minimized.fun,
    #             "Labor": minimized.x[0],
    #             "Capital": minimized.x[1]
    #         })
    #
    # vals = pd.DataFrame(dict)

    # vals = pd.read_pickle("price_wage_df.pkl")

    # fig = plt.figure()
    #
    # ax = fig.add_subplot(1, 1, 1, projection="3d")
    # ax.plot_surface(vals["Price"].to_numpy().reshape(100, 100), vals["Wage"].to_numpy().reshape(100, 100), vals["Labor"].to_numpy().reshape(100, 100), cmap="viridis")
    # ax.set_xlabel("Price")
    # ax.set_ylabel("Wage")
    # ax.set_zlabel("Labor Demand")
    #
    #
    # plt.show()

    labor_capital = np.meshgrid(np.linspace(0.1, 10, 100), np.linspace(0.1, 10, 100), indexing="ij")
    productions = production_function(labor_capital, A, alpha, time)
    profits = profit_function(labor_capital, A, alpha, time, wage, rate, price)
    minimized, production = get_max_profit(A, alpha, time, wage, rate, price)
    fig = plt.figure()

    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(labor_capital[0], labor_capital[1], productions, cmap="viridis")
    ax.set_xlabel("Labor")
    ax.set_ylabel("Capital")
    ax.set_zlabel("Production")

    ax = fig.add_subplot(122, projection="3d")
    ax.plot_surface(labor_capital[0], labor_capital[1], profits, cmap="viridis")
    ax.scatter(minimized.x[0], minimized.x[1], -minimized.fun, color="red")
    ax.set_xlabel("Labor")
    ax.set_ylabel("Capital")
    ax.set_zlabel("Profit")
    plt.show()

    # ax = fig.add_subplot(133, projection="3d")

    pass

if __name__ == "__main__":
    main()
