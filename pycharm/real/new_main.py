from scipy.optimize import differential_evolution
import firm
import consumer
import constants
from typing import Callable

times_ran = 0

def costs(variables: tuple[float, float, float],
          constants: tuple[float, float, float],
          good_supply: Callable[[float, float, float, float, float, float], float],
          good_demand: Callable[[float, float], float],
          capital_supply: Callable[[float, float, float, float, float, float], float],
          labor_supply: Callable[[float, float], float],
          demands_func: Callable[[float, float, float, float, float, float], tuple[float, float, float]]) -> (float, (float, float, float), (float, float, float, float, float, float)):

    price, wage, rate_k = variables
    A, time, rate_z = constants

    g_sup = good_supply(time, A, price, wage, rate_k, rate_z)
    g_dem = good_demand(price, wage)
    good_cost = g_sup - g_dem

    labor_demand, capital_demand, _ = demands_func(time, A, price, wage, rate_k, rate_z)

    l_sup = labor_supply(price, wage)
    labor_cost = l_sup - labor_demand

    c_sup = capital_supply(time, A, price, wage, rate_k, rate_z)
    capital_cost = c_sup - capital_demand

    total_cost = (good_cost**2) + (labor_cost**2) + (capital_cost**2)
    global times_ran

    # print(f"\t\tTimes Ran: {times_ran}, Total Cost: {total_cost}, Good Cost: {good_cost}, Labor Cost: {labor_cost}, Capital Cost: {capital_cost}, Good Supply: {g_sup}, Good Demand: {g_dem}, Labor Supply: {l_sup}, Labor Demand: {labor_demand}, Capital Supply: {c_sup}, Capital Demand: {capital_demand}")
    times_ran += 1
    return total_cost, (good_cost, labor_cost, capital_cost), (g_sup, g_dem, l_sup, labor_demand, c_sup, capital_demand)


def main():
    price = 78.07508699607666
    wage = 87.65360897225864
    rate_k = 39.93574375702552

    good_firm = firm.Firm()
    capital_firm = firm.Firm()

    consumer_1 = consumer.Consumer()

    good_supply = lambda time, A, price, wage, rate_k, rate_z: good_firm.max_profit(time, A, price, wage, rate_k, rate_z)[1]
    capital_supply = lambda time, A, price, wage, rate_k, rate_z: capital_firm.max_profit(time, A, price, wage, rate_k, rate_z)[1]

    labor_supply = lambda price, wage: consumer.Consumer.get_total_work_demand(price, wage)
    # issue: get_total_time_demand returns total free time demanded, not total free time offered.
    # TODO: set max_free_time to 24*7, then get_total_time_demand = max_free_time - get_total_time_demand
    good_demand = lambda price, wage: consumer.Consumer.get_total_good_demand(price, wage)

    demands_func = lambda time, A, price, wage, rate_k, rate_z: firm.Firm.get_total_demands(time, A, price, wage, rate_k, rate_z)

    # total_cost, cost, vals = costs((price, wage, rate_k), (A, time, rate_z), good_supply, good_demand, capital_supply, labor_supply, demands_func)
    #
    # print(f"Total Cost: {total_cost}, Good Cost: {cost[0]}, Labor Cost: {cost[1]}, Capital Cost: {cost[2]}, Good Supply: {vals[0]}, Good Demand: {vals[1]}, Labor Supply: {vals[2]}, Labor Demand: {vals[3]}, Capital Supply: {vals[4]}, Capital Demand: {vals[5]}")

    foo = differential_evolution(lambda x: costs(x, (constants.A, constants.time, constants.rate_z), good_supply, good_demand, capital_supply, labor_supply, demands_func)[0], bounds=[(0.2, 1000), (0.2, 1000), (0.2, 1000)], maxiter=100, tol=0.1, disp=True)

    print(foo)
    bar = good_firm.max_profit(constants.time, constants.A, foo.x[0], foo.x[1], foo.x[2], constants.rate_z)
    print([i for i in bar])
    print("pass")


if __name__ == "__main__":
    main()