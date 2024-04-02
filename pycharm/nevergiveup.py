import numpy as np
from scipy.optimize import fsolve, minimize, OptimizeResult, least_squares, brute
import matplotlib.pyplot as plt
import pandas as pd

def production_function(variables: tuple[float, float, float], constants: tuple[float, float, float, float], time: float, A: float) -> float:
    labor, capital, robots = variables
    alpha_1, alpha_2, eos_1, eos_2 = constants
    capital_term = (alpha_1 ** (1/eos_1)) * (capital ** (((-capital/time)+np.e)*((eos_1-1)/eos_1)))
    labor_inner_term = (alpha_2 ** (1/eos_2)) * (labor ** (((-labor/time)+np.e)*((eos_2-1)/eos_2)))
    robot_inner_term = ((1-alpha_2) ** (1/eos_2)) * (robots ** (((-robots/time)+np.e)*((eos_2-1)/eos_2)))
    labor_outer_term = ((1-alpha_1) ** (1/eos_1)) * ((labor_inner_term + robot_inner_term) ** ((eos_2/(eos_2-1)) * (eos_1-1)/eos_1))
    final_term = A *((capital_term + labor_outer_term) ** (eos_1/(eos_1-1)))
    return final_term

def profit_function(variables: tuple[float, float, float], constants: tuple[float, float, float, float],
                    time: float, A: float, wage: float, rate_k: float, rate_z: float, price: float) -> float:
    labor, capital, robots = variables
    # alpha_1, alpha_2, eos_1, eos_2, time, A = constants
    revenue = price * production_function(variables, constants, time, A)
    costs = (wage * labor) + (rate_k * capital) + (rate_z * robots)
    profit = revenue - costs
    # print(f"revenue: {revenue}, costs: {costs}, profit: {profit}")
    return profit

def get_max_profit(constants: tuple[float, float, float, float], time: float, A: float, wage: float,
                   rate_k: float, rate_z: float, price: float) -> (OptimizeResult, float):
    minimized = minimize(lambda x: -1 * profit_function(x, constants, time, A, wage, rate_k, rate_z, price),
                         np.array([10, 10, 10]), bounds=[(0.2, None), (0.2, None), (0.2, None)])
    return minimized, production_function(minimized.x, constants, time, A)

def product_demand(price: float) -> float:
    return 10 - (0.5*price)

def labor_supply(wage: float) -> float:
    return -1 + wage**1.2

def equilibrium_curve_2(variables: tuple[float, float], constants: tuple[float, float, float, float],
                        A: float, time: float, rate_k: float, rate_z: float) -> list[float, float]:
    price, wage = variables
    max_profit, production = get_max_profit(constants, time, A, wage, rate_k, rate_z, price)
    quantity_curve = (production - product_demand(price)) ** 2
    labor_curve = (labor_supply(wage) - max_profit.x[0]) ** 2
    print(f"Price: {price}, Wage: {wage}, Quantity: {quantity_curve}, Labor: {labor_curve}")
    return [quantity_curve, labor_curve]

def equilibrium_curve_test(variables: tuple[float, float], constants: tuple[float, float, float, float],
                           A: float, time: float, rate_k: float, rate_z: float) -> float:
    price, wage = variables
    max_profit, production = get_max_profit(constants, A, time, wage, rate_k, rate_z, price)
    quantity_curve = (production - product_demand(price)) ** 2
    labor_curve = (labor_supply(wage) - max_profit.x[0]) ** 2
    # print(f"Price: {price}, Wage: {wage}, Quantity: {quantity_curve}, Labor: {labor_curve}, Cost: {quantity_curve + labor_curve}")
    return quantity_curve + labor_curve

def find_equilibrium_2(constants: tuple[float, float, float, float], A: float,
                       time: float, rate_k: float, rate_z: float) -> OptimizeResult:
    return least_squares(equilibrium_curve_2, np.array([10, 10]), args=(constants, A, time, rate_k, rate_z),
                         bounds=[(0.2, 0.2), (100, 100)], xtol=1e-10, ftol=1e-10, gtol=1e-10)

# def equilibrium_curve(price, A, alpha, time, wage, rate):
#     max_profit, production = get_max_profit(A, alpha, time, wage, rate, price)
#     point =  production - product_demand(price)
#     # print(f"Price: {price}, Point: {point}")
#     return point

# def find_equilibrium(A, alpha, time, wage, rate):
#     return fsolve(equilibrium_curve, np.array([10]), args=(A, alpha, time, wage, rate))

def main():
    A = 5
    alpha_1 = 0.75
    alpha_2 = 0.75
    eos_1 = 0.75
    eos_2 = 0.75
    constants = (alpha_1, alpha_2, eos_1, eos_2)
    time = 1
    wage = 1.2
    rate_k = 6
    rate_z = 3
    price = 2.5

    # foo = find_equilibrium_2(constants, A, time, rate_k, rate_z)
    # print("\n\n\n\n\n")
    # bar = minimize(equilibrium_curve_test, np.array([10, 10]), bounds=((0.2, None), (0.2, None)), args=(constants, A, time, rate_k, rate_z))
    # return foo, bar
    ans = brute(equilibrium_curve_test, (slice(0.2, 15, 0.01), slice(0.2, 15, 0.01)), args=(constants, A, time, rate_k, rate_z), full_output=False, finish=None, workers=1)
    print(ans)
    return ans

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

    # price_wage = np.meshgrid(np.linspace(1, 20, 100), np.linspace(1, 20, 100))
    # costs = np.full_like(price_wage[0], fill_value=None, dtype=np.dtype)
    # q_costs = np.full_like(price_wage[0], fill_value=None, dtype=np.dtype)
    # l_costs = np.full_like(price_wage[0], fill_value=None, dtype=np.dtype)
    # for i in range(len(price_wage[0])):
    #     for j in range(len(price_wage[1][i])):
    #         costs[i][j] = equilibrium_curve_test((price_wage[0][i][j], price_wage[1][i][j]), constants, A, time, rate_k, rate_z)
    #         q_costs[i][j], l_costs[i][j] = equilibrium_curve_2((price_wage[0][i][j], price_wage[1][i][j]), constants, A, time, rate_k, rate_z)
    #
    # fig = plt.figure()
    #
    # ax = fig.add_subplot(1, 3, 1, projection="3d")
    # ax.plot_surface(price_wage[0], price_wage[1], costs, cmap="viridis")
    # ax.set_xlabel("Price")
    # ax.set_ylabel("Wage")
    # ax.set_zlabel("Cost")
    #
    # ax = fig.add_subplot(1, 3, 2, projection="3d")
    # ax.plot_surface(price_wage[0], price_wage[1], q_costs, cmap="viridis")
    # ax.set_xlabel("Price")
    # ax.set_ylabel("Wage")
    # ax.set_zlabel("Quantity Cost")
    #
    # ax = fig.add_subplot(1, 3, 3, projection="3d")
    # ax.plot_surface(price_wage[0], price_wage[1], l_costs, cmap="viridis")
    # ax.set_xlabel("Price")
    # ax.set_ylabel("Wage")
    # ax.set_zlabel("Labor Cost")
    #
    # plt.show()
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

    # labor_capital = np.meshgrid(np.linspace(0.1, 10, 100), np.linspace(0.1, 10, 100), indexing="ij")
    # productions = production_function(labor_capital, A, alpha, time)
    # profits = profit_function(labor_capital, A, alpha, time, wage, rate, price)
    # minimized, production = get_max_profit(A, alpha, time, wage, rate, price)
    # fig = plt.figure()
    #
    # ax = fig.add_subplot(121, projection="3d")
    # ax.plot_surface(labor_capital[0], labor_capital[1], productions, cmap="viridis")
    # ax.set_xlabel("Labor")
    # ax.set_ylabel("Capital")
    # ax.set_zlabel("Production")
    #
    # ax = fig.add_subplot(122, projection="3d")
    # ax.plot_surface(labor_capital[0], labor_capital[1], profits, cmap="viridis")
    # ax.scatter(minimized.x[0], minimized.x[1], -minimized.fun, color="red")
    # ax.set_xlabel("Labor")
    # ax.set_ylabel("Capital")
    # ax.set_zlabel("Profit")
    # plt.show()

    # ax = fig.add_subplot(133, projection="3d")

    pass

if __name__ == "__main__":
    main()
