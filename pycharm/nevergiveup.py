import numpy as np
from scipy.optimize import fsolve, minimize, OptimizeResult
import matplotlib.pyplot as plt


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
    return 5 + wage**1.2

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
    rate = 0.8
    price = 2.5

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

    inputs = np.linspace(1, 7.5, 100)
    optimized_results = []
    demand_curve = product_demand(inputs)
    for i in inputs:
        optimized_results.append(get_max_profit(A, alpha, time, wage, rate, i))
        equilibrium_curve(i, A, alpha, time, wage, rate)

    fig = plt.figure()

    ax = fig.add_subplot(4, 1, 1)
    ax.plot(inputs, np.array([-i[0].fun for i in optimized_results]), color="red")
    # ax.plot(inputs, demand_curve, color="blue")
    # ax.plot(inputs, np.array([-result[0].fun - demand_curve[i] for i, result in enumerate(optimized_results)]), color="green")
    # ax.plot(inputs, np.array([0 for i in inputs]), color="black")
    ax.scatter(find_equilibrium(A, alpha, time, wage, rate), 0, color="green")
    ax.set_xlabel("Price")
    ax.set_ylabel("Profit")

    ax = fig.add_subplot(4, 1, 2)
    ax.plot(inputs, np.array([result[1] for i, result in enumerate(optimized_results)]), color="red")
    ax.plot(inputs, demand_curve, color="blue")
    # ax.plot(inputs, np.array([result[1]-demand_curve[i] for i, result in enumerate(optimized_results)]), color="green")
    # ax.plot(inputs, np.array([0 for i in inputs]), color="black")
    ax.scatter(find_equilibrium(A, alpha, time, wage, rate), product_demand(find_equilibrium(A, alpha, time, wage, rate)), color="green")
    ax.set_xlabel("Price")
    ax.set_ylabel("Quantity Produced")

    ax = fig.add_subplot(4, 1, 3)
    ax.plot(inputs, np.array([i[0].x[0] for i in optimized_results]), color="blue")
    ax.set_xlabel("Price")
    ax.set_ylabel("Labor")

    ax = fig.add_subplot(4, 1, 4)
    ax.plot(inputs, np.array([i[0].x[1] for i in optimized_results]), color="blue")
    ax.set_xlabel("Price")
    ax.set_ylabel("Capital")

    plt.show()

if __name__ == "__main__":
    main()
