import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Consumer:

    instances = {}
    max_time = 5

    def __init__(self):
        self.alpha = 0.75 # alpha close to 1 -> leisure is not very important (work a lot), alpha close to 0 -> leisure is very important (work less)
        self.income = 120 # higher income -> less importance of working
        # https://open.lib.umn.edu/principleseconomics/chapter/12-2-the-supply-of-labor/
        # https://saylordotorg.github.io/text_introduction-to-economic-analysis/s14-01-labor-supply.html
        Consumer.instances[self] = self

    def utility_func(self, q_good: float, q_work: float) -> float:
        return (q_good**self.alpha) * (q_work**(1-self.alpha))

    def good_demand(self, cost_good: float, cost_work: float) -> float: # cost_good = price of good, cost_work = -wage
        return (self.income + (cost_work * self.max_time)) / (cost_good * (1 + ((1-self.alpha) / self.alpha)))

    def work_demand(self, cost_good: float, cost_work: float) -> float: # cost_good = price of good, cost_work = -wage
        # individual work_demand == individual labor_supply
        return np.fmax(0., (self.alpha * self.max_time) - ((1-self.alpha) * (self.income / cost_work)))

    @classmethod
    def get_total_good_demand(cls, cost_good: float, cost_work: float) -> float: # cost_good = price of good, cost_work = -wage
        return sum([cls.instances[instance].good_demand(cost_good, cost_work) for instance in cls.instances])

    @classmethod
    def get_total_work_demand(cls, cost_good: float, cost_work: float) -> float: # cost_good = price of good, cost_work = -wage
        # total work_demand == total labor_supply
        return sum([cls.instances[instance].work_demand(cost_good, cost_work) for instance in cls.instances])

    @classmethod
    def get_total_utility(cls, q_good: float, q_work: float) -> float:
        return sum([cls.instances[instance].utility_func(q_good, q_work) for instance in cls.instances])

def main():
    consumer1 = Consumer()
    # consumer2 = Consumer()
    # print(f"Good Weight 1: {consumer1.good_weight}, Time Weight 1: {consumer1.work_weight}, Income 1: {consumer1.income}")
    # print(f"Good Weight 2: {consumer2.good_weight}, Time Weight 2: {consumer2.work_weight}, Income 2: {consumer2.income}")

    hold_price = np.linspace(5,5,100)

    wage_range = np.linspace(0, 25, 100)
    income_range = np.linspace(0.05, 0.95, 4)
    # df = pd.DataFrame(columns=["Alpha", "Wage", "Work Demand"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for alpha in income_range:
        consumer1.alpha = alpha
        work_demand = consumer1.work_demand(hold_price, wage_range)
        # good_demand = consumer1.good_demand(wage_range, hold_price)
        # df.loc[len(df)] = [alpha, wage_range, work_demand]
        ax.plot(wage_range, work_demand, label=f"Work Alpha: {alpha}")
        # ax.plot(wage_range, good_demand, label=f"Good Alpha: {alpha}")
    ax.set_xlabel("Wage")
    ax.set_ylabel("Quantity Labor")
    ax.legend()

    plt.show()

    # G, T = np.meshgrid(np.linspace(1, 10, 100), np.linspace(1, 10, 100))
    #
    # U1 = consumer1.utility_func(G, T)
    # g1_demand = consumer1.good_demand(G, T)
    # w1_demand = consumer1.work_demand(G, T)
    #
    # U2 = consumer2.utility_func(G, T)
    # g2_demand = consumer2.good_demand(G, T)
    # w2_demand = consumer2.work_demand(G, T)
    #
    # UT = Consumer.get_total_utility(G, T)
    # gT_demand = Consumer.get_total_good_demand(G, T)
    # wT_demand = Consumer.get_total_work_demand(G, T)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(331, projection='3d')
    # ax.plot_surface(G, T, U1, cmap="viridis")
    # ax.set_xlabel("Good")
    # ax.set_ylabel("Time")
    # ax.set_zlabel("Utility 1")
    #
    # ax = fig.add_subplot(332, projection='3d')
    # ax.plot_surface(G, T, g1_demand, cmap="viridis")
    # ax.set_xlabel("Cost Good")
    # ax.set_ylabel("Cost Time")
    # ax.set_zlabel("Good Demand 1")
    #
    # ax = fig.add_subplot(333, projection='3d')
    # ax.plot_surface(G, T, w1_demand, cmap="viridis")
    # ax.set_xlabel("Cost Good")
    # ax.set_ylabel("Cost Time")
    # ax.set_zlabel("Time Demand 1")
    #
    # ax = fig.add_subplot(334, projection='3d')
    # ax.plot_surface(G, T, U2, cmap="viridis")
    # ax.set_xlabel("Good")
    # ax.set_ylabel("Time")
    # ax.set_zlabel("Utility 2")
    #
    # ax = fig.add_subplot(335, projection='3d')
    # ax.plot_surface(G, T, g2_demand, cmap="viridis")
    # ax.set_xlabel("Cost Good")
    # ax.set_ylabel("Cost Time")
    # ax.set_zlabel("Good Demand 2")
    #
    # ax = fig.add_subplot(336, projection='3d')
    # ax.plot_surface(G, T, w2_demand, cmap="viridis")
    # ax.set_xlabel("Cost Good")
    # ax.set_ylabel("Cost Time")
    # ax.set_zlabel("Time Demand 2")
    #
    # ax = fig.add_subplot(337, projection='3d')
    # ax.plot_surface(G, T, UT, cmap="viridis")
    # ax.set_xlabel("Good")
    # ax.set_ylabel("Time")
    # ax.set_zlabel("Utility T")
    #
    # ax = fig.add_subplot(338, projection='3d')
    # ax.plot_surface(G, T, gT_demand, cmap="viridis")
    # ax.set_xlabel("Cost Good")
    # ax.set_ylabel("Cost Time")
    # ax.set_zlabel("Good Demand T")
    #
    # ax = fig.add_subplot(339, projection='3d')
    # ax.plot_surface(G, T, wT_demand, cmap="viridis")
    # ax.set_xlabel("Cost Good")
    # ax.set_ylabel("Cost Time")
    # ax.set_zlabel("Time Demand T")
    #
    # plt.show()

if __name__ == "__main__":
    main()