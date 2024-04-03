from scipy.optimize import differential_evolution, OptimizeResult
import firm
import consumer
import constants
import new_graphs
from typing import Callable, Sequence
import matplotlib.pyplot as plt
import numpy as np

# EQUILIBRIUM IS IDEAL FOR THE MONOPOLISTIC FIRM BC GOOD'S SUPPLY CURVE (WHICH HAS MAXIMIZED PROFIT; i.e. ANY OTHER PRODUCTION QUANTITY IS INEFFICIENT)
# IS EQUAL TO THE GOOD'S DEMAND CURVE, SO EVERYTHING PRODUCED WILL BE SOLD (UNDER IDEALIZED PROFIT CONDITION)


def costs(variables: tuple[float, float],
          constant: tuple[float, float, float, float],
          good_supply: Callable[[float, float, float, float, float, float], float],
          good_demand: Callable[[float, float], float],
          labor_supply: Callable[[float, float], float],
          demands_func: Callable[[float, float, float, float, float, float], float]) -> (float, (float, float, float), (float, float, float, float, float, float)):

    price, wage = variables
    A, time, rate_k, rate_z = constant

    g_sup = good_supply(time, A, price, wage, rate_k, rate_z)
    g_dem = good_demand(price, wage)
    good_cost = g_sup - g_dem

    labor_demand = demands_func(time, A, price, wage, rate_k, rate_z)

    l_sup = labor_supply(price, wage)
    labor_cost = l_sup - labor_demand

    total_cost = (good_cost**2) + (labor_cost**2)

    # print(f"\t\tTimes Ran: {times_ran}, Total Cost: {total_cost}, Good Cost: {good_cost}, Labor Cost: {labor_cost}, Capital Cost: {capital_cost}, Good Supply: {g_sup}, Good Demand: {g_dem}, Labor Supply: {l_sup}, Labor Demand: {labor_demand}, Capital Supply: {c_sup}, Capital Demand: {capital_demand}")
    return total_cost, (good_cost, labor_cost), (g_sup, g_dem, l_sup, labor_demand)

def find_equilibrium(firms: Sequence[firm.Firm], consumers: Sequence[consumer.Consumer],
                     constant: tuple[float, float, float, float]) -> OptimizeResult:
    good_supply = lambda time, A, price, wage, rate_k, rate_z: sum([iter_firm.max_profit(time, A, price, wage, rate_k, rate_z)[1] for iter_firm in firms])
    good_demand = lambda price, wage: sum([iter_consumer.good_demand(price, wage) for iter_consumer in consumers])

    labor_supply = lambda price, wage: sum([iter_consumer.work_demand(price, wage) for iter_consumer in consumers])
    labor_demand = lambda time, A, price, wage, rate_k, rate_z: sum([iter_firm.max_profit(time, A, price, wage, rate_k, rate_z)[0].x[0] for iter_firm in firms])

    return differential_evolution(lambda x: costs(x, constant, good_supply, good_demand, labor_supply, labor_demand)[0],
                                   bounds=[(0.2, 100), (0.2, 100)], maxiter=100, atol=0.01, disp=True, polish=False)

def main():

    good_firm = firm.Firm()

    consumer_1 = consumer.Consumer()

    # good_supply = lambda time, A, price, wage, rate_k, rate_z: good_firm.max_profit(time, A, price, wage, rate_k, rate_z)[1]

    # labor_supply = lambda price, wage: consumer.Consumer.get_total_work_demand(price, wage)
    # good_demand = lambda price, wage: consumer.Consumer.get_total_good_demand(price, wage)

    # labor_demand = lambda time, A, price, wage, rate_k, rate_z: firm.Firm.get_total_demands(time, A, price, wage, rate_k, rate_z)[0]

    # total_cost, cost, vals = costs((price, wage, rate_k), (A, time, rate_z), good_supply, good_demand, capital_supply, labor_supply, demands_func)
    #
    # print(f"Total Cost: {total_cost}, Good Cost: {cost[0]}, Labor Cost: {cost[1]}, Capital Cost: {cost[2]}, Good Supply: {vals[0]}, Good Demand: {vals[1]}, Labor Supply: {vals[2]}, Labor Demand: {vals[3]}, Capital Supply: {vals[4]}, Capital Demand: {vals[5]}")

    # foo = differential_evolution(lambda x: costs(x, (constants.A, constants.time, constants.rate_k, constants.rate_z),
    #                                              good_supply, good_demand, labor_supply, labor_demand)[0],
    #                              bounds=[(0.2, 100), (0.2, 100)], maxiter=100, atol=0.01, disp=True, polish=False)


    # print(foo)

    foo = find_equilibrium([good_firm], [consumer_1], (constants.A, constants.time, constants.rate_k, constants.rate_z))
    new_graphs.draw_graphs([good_firm], [consumer_1], foo.x[0], foo.x[1])
    print("pass")


if __name__ == "__main__":
    main()