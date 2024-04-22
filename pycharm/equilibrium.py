from scipy.optimize import OptimizeResult, minimize
import firm
import consumer
import new_graphs
from typing import Callable, Sequence
import numpy as np

# EQUILIBRIUM IS IDEAL FOR THE MONOPOLISTIC FIRM BC GOOD'S SUPPLY CURVE (WHICH HAS MAXIMIZED PROFIT; i.e. ANY OTHER PRODUCTION QUANTITY IS INEFFICIENT)
# IS EQUAL TO THE GOOD'S DEMAND CURVE, SO EVERYTHING PRODUCED WILL BE SOLD (UNDER IDEALIZED PROFIT CONDITION)

def costs(variables: tuple[float, float],
          constant: tuple[float, float, float, float],
          good_supply: Callable[[tuple, float, float], float],
          good_demand: Callable[[float, float], float],
          labor_supply: Callable[[float, float], float],
          demands_func: Callable[[tuple, float, float], float]) -> (float, (float, float, float), (float, float, float, float, float, float)):

    price, wage = variables
    # print(f"eval at {price}, {wage}")
    A, time, rate_k, rate_z = constant

    g_sup = good_supply(constant, price, wage)
    g_dem = good_demand(price, wage)
    good_cost = g_sup - g_dem

    labor_demand = demands_func(constant, price, wage)

    l_sup = labor_supply(price, wage)
    labor_cost = l_sup - labor_demand

    total_cost = (good_cost**2) + (labor_cost**2)

    # print(f"\t\tTimes Ran: {times_ran}, Total Cost: {total_cost}, Good Cost: {good_cost}, Labor Cost: {labor_cost}, Capital Cost: {capital_cost}, Good Supply: {g_sup}, Good Demand: {g_dem}, Labor Supply: {l_sup}, Labor Demand: {labor_demand}, Capital Supply: {c_sup}, Capital Demand: {capital_demand}")
    return total_cost, (good_cost, labor_cost), (g_sup, g_dem, l_sup, labor_demand)

def find_equilibrium(firms: Sequence[firm.Firm], consumers: Sequence[consumer.Consumer],
                     constant: tuple[float, float, float, float, float, float], max_trials=1) -> OptimizeResult:
    good_supply = lambda constant_vars, price, wage: sum([iter_firm.max_profit(constant_vars, price, wage, constant[5])[1] for iter_firm in firms])
    good_demand = lambda price, wage: sum([iter_consumer.good_demand(price, wage, constant[4]) for iter_consumer in consumers])

    labor_supply = lambda price, wage: sum([iter_consumer.work_demand(price, wage, constant[4]) for iter_consumer in consumers])
    labor_demand = lambda constant_vars, price, wage: sum([iter_firm.max_profit(constant_vars, price, wage, constant[5])[0].x[0] for iter_firm in firms])
    # print(f"constant[:3]: {constant[:3]}, constant[:4]: {constant[:4]}, constant[4]: {constant[4]}")


    # P, W = np.meshgrid(np.linspace(0.15,2, 50), np.linspace(0.5, 2, 25))
    # Z = np.zeros_like(P)
    # print("starting function calls")
    # for i in range(P.shape[0]):
    #     for j in range(P.shape[1]):
    #         Z[i, j] = costs((P[i, j], W[i, j]), constant[:4], good_supply, good_demand, labor_supply, labor_demand)[0]
    # print("finished function calls, graphing...")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(P, W, Z)
    # plt.show()

    if max_trials < 1:
        max_trials = 1
    trials = 0
    equi = []
    while trials < max_trials:
        equi.append(minimize(lambda x: costs(x, constant[:4], good_supply, good_demand, labor_supply, labor_demand)[0],
                        np.array([float(trials+0.1), float(trials+0.1)]), method="Nelder-Mead", bounds=[(0.0001, None), (0.0001, None)], options={"eps": 0.0001}))
        print(f"minimization number {trials+1} complete at ({equi[-1].x}): {equi[-1].fun}")
        # equi = differential_evolution(lambda x: costs(x, constant, good_supply, good_demand, labor_supply, labor_demand)[0],
        #                            bounds=[(0.0001, 100), (0.0001, 20)], maxiter=200, atol=0.1, disp=True, polish=True)
        # bnds = ((0.0001, 100), (0.0001, 100))
        # bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bnds]), np.array([b[1] for b in bnds]))
        # minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bnds, "options": {"eps": 0.001}}
        # equi = basinhopping(lambda x: costs(x, constant, good_supply, good_demand, labor_supply, labor_demand)[0],
        #                     np.array([0.1, 0.1]), niter=20, minimizer_kwargs=minimizer_kwargs, disp=True, take_step=bounded_step)
        trials += 1
        if equi[-1].fun < 0.1:
            break
    return sorted(equi, key=lambda x: x.fun)[0]

def main():

    good_firm = firm.Firm()

    upper = consumer.Consumer("upper", 50)
    middle = consumer.Consumer("middle", 10)
    lower = consumer.Consumer("lower", 5)

    # good_supply = lambda constant, price, wage: good_firm.max_profit(constant, price, wage[1]

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


    foo = find_equilibrium([good_firm], [lower, middle, upper], (constants.A, constants.time, constants.rate_k, constants.rate_z), 5)
    new_graphs.draw_graphs([good_firm], [lower, middle, upper], foo.x[0], foo.x[1])
    print(f"price: {foo.x[0]}, wage: {foo.x[1]}")
    # new_graphs.draw_graphs([good_firm], [consumer_1], 0.08, 0.35)


if __name__ == "__main__":
    main()