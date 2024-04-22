import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

class Consumer:

    instances = {}
    max_time = 1 # maximum amount of hours you can work

    def __init__(self, name="consumer", non_wage_income=50, p_leisure=0.6, p_good=0.35, p_save=0.05):
        self.name = name
        self.non_wage_income = non_wage_income
        # higher non_wage_income -> less importance of working
        # https://open.lib.umn.edu/principleseconomics/chapter/12-2-the-supply-of-labor/
        # https://saylordotorg.github.io/text_introduction-to-economic-analysis/s14-01-labor-supply.html


        self.p_leisure = p_leisure
        self.p_good = p_good
        self.p_save = p_save

        # preferences must add up to 1



        self.num_goods = 0
        self.money_saved = 0
        self.leisure = self.max_time
        self.time_working = 0
        self.util = 0

        self.demands = (0, 0, 0)

        self.money = self.non_wage_income
        self.starting_money = self.non_wage_income

        Consumer.instances[self] = self


    def utility_func(self, q_good: float, q_leisure: float, q_save: float) -> float:
        return (self.p_good * (q_good ** 0.5)) + (self.p_leisure * (q_leisure ** 0.5)) + (self.p_save * (q_save ** 0.5))

    def set_util(self):
        self.util = self.utility_func(self.num_goods, self.leisure, self.money_saved)

    def good_demand(self, cost_good: float, cost_work: float, cost_save: float) -> float:
        # returns quantity of good willing to buy
        return ((self.money + (self.max_time * cost_work)) /
                (cost_good * (1 + (((self.p_leisure / self.p_good) ** 2) * (cost_good / cost_work)) +
                              (((self.p_save / self.p_good) ** 2) * (cost_good / cost_save)))))

    def leisure_demand(self, cost_good: float, cost_work: float, cost_save: float) -> float: # cost_work = wage
        # returns leisure wanted
        return ((self.money + (self.max_time * cost_work)) /
                (cost_work * (1 + (((self.p_good / self.p_leisure) ** 2) * (cost_work / cost_good)) +
                (((self.p_save / self.p_leisure) ** 2) * (cost_work / cost_save)))))

    def work_demand(self, cost_good: float, cost_work: float, cost_save: float) -> float:
        return np.fmax(0., self.max_time - self.leisure_demand(cost_good, cost_work, cost_save))

    def save_demand(self, cost_good: float, cost_work: float, cost_save: float) -> float:
        return ((self.money + (self.max_time * cost_work)) /
                (cost_save * (1 + (((self.p_good / self.p_save) ** 2) * (cost_save / cost_good)) +
                              (((self.p_leisure / self.p_save) ** 2) * (cost_save / cost_work)))))

    def get_demands(self, cost_good: float, cost_work: float, cost_save: float) -> (float, float, float):
        return (self.good_demand(cost_good, cost_work, cost_save),
                self.work_demand(cost_good, cost_work, cost_save),
                self.save_demand(cost_good, cost_work, cost_save))

    def purchase(self, quantity: float, price: float):
        self.money -= quantity * price
        self.num_goods += quantity

    def work(self, hours: float, wage: float):
        self.money += hours * wage
        self.time_working += hours
        self.leisure -= hours

    def save(self, save: float):
        self.money -= save
        self.money_saved += save

    def max_purchase(self, price: float) -> float:
        return self.money / price

    def next_year(self, i_r: float):
        self.starting_money = self.non_wage_income + (self.money_saved * (1 + i_r))
        self.money = self.starting_money
        self.num_goods = 0
        self.money_saved = 0
        self.leisure = self.max_time
        self.time_working = 0
        self.util = 0

    def restart(self):
        self.money = self.non_wage_income
        self.num_goods = 0
        self.money_saved = 0
        self.leisure = self.max_time
        self.time_working = 0
        self.util = 0



def old_graphs():
    consumer1 = Consumer()
    # consumer2 = Consumer()
    # print(f"Good Weight 1: {consumer1.good_weight}, Time Weight 1: {consumer1.work_weight}, Income 1: {consumer1.non_wage_income}")
    # print(f"Good Weight 2: {consumer2.good_weight}, Time Weight 2: {consumer2.work_weight}, Income 2: {consumer2.non_wage_income}")

    hold_price = np.linspace(5,5,100)

    wage_range = np.linspace(0, 25, 100)
    income_range = np.linspace(0.05, 0.95, 4)
    # df = pd.DataFrame(columns=["Alpha", "Wage", "Work Demand"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for alpha in income_range:
        consumer1.alpha = alpha
        work_demand = consumer1.leisure_demand(hold_price, wage_range)
        # good_demand = consumer1.good_demand(wage_range, hold_price)
        # df.loc[len(df)] = [alpha, wage_range, work_demand]
        ax.plot(wage_range, work_demand, label=f"Work Alpha: {alpha}")
        # ax.plot(wage_range, good_demand, label=f"Good Alpha: {alpha}")
    ax.set_xlabel("Wage")
    ax.set_ylabel("Quantity Labor")
    ax.legend()

    plt.show()

def older_graphs():
    consumer1 = Consumer()
    consumer2 = Consumer()

    G, T = np.meshgrid(np.linspace(1, 10, 100), np.linspace(1, 10, 100))

    U1 = consumer1.utility_func(G, T)
    g1_demand = consumer1.good_demand(G, T)
    w1_demand = consumer1.leisure_demand(G, T)

    U2 = consumer2.utility_func(G, T)
    g2_demand = consumer2.good_demand(G, T)
    w2_demand = consumer2.leisure_demand(G, T)

    UT = Consumer.get_total_utility(G, T)
    gT_demand = Consumer.get_total_good_demand(G, T)
    wT_demand = Consumer.get_total_work_demand(G, T)

    fig = plt.figure()
    ax = fig.add_subplot(331, projection='3d')
    ax.plot_surface(G, T, U1, cmap="viridis")
    ax.set_xlabel("Good")
    ax.set_ylabel("Time")
    ax.set_zlabel("Utility 1")

    ax = fig.add_subplot(332, projection='3d')
    ax.plot_surface(G, T, g1_demand, cmap="viridis")
    ax.set_xlabel("Cost Good")
    ax.set_ylabel("Cost Time")
    ax.set_zlabel("Good Demand 1")

    ax = fig.add_subplot(333, projection='3d')
    ax.plot_surface(G, T, w1_demand, cmap="viridis")
    ax.set_xlabel("Cost Good")
    ax.set_ylabel("Cost Time")
    ax.set_zlabel("Time Demand 1")

    ax = fig.add_subplot(334, projection='3d')
    ax.plot_surface(G, T, U2, cmap="viridis")
    ax.set_xlabel("Good")
    ax.set_ylabel("Time")
    ax.set_zlabel("Utility 2")

    ax = fig.add_subplot(335, projection='3d')
    ax.plot_surface(G, T, g2_demand, cmap="viridis")
    ax.set_xlabel("Cost Good")
    ax.set_ylabel("Cost Time")
    ax.set_zlabel("Good Demand 2")

    ax = fig.add_subplot(336, projection='3d')
    ax.plot_surface(G, T, w2_demand, cmap="viridis")
    ax.set_xlabel("Cost Good")
    ax.set_ylabel("Cost Time")
    ax.set_zlabel("Time Demand 2")

    ax = fig.add_subplot(337, projection='3d')
    ax.plot_surface(G, T, UT, cmap="viridis")
    ax.set_xlabel("Good")
    ax.set_ylabel("Time")
    ax.set_zlabel("Utility T")

    ax = fig.add_subplot(338, projection='3d')
    ax.plot_surface(G, T, gT_demand, cmap="viridis")
    ax.set_xlabel("Cost Good")
    ax.set_ylabel("Cost Time")
    ax.set_zlabel("Good Demand T")

    ax = fig.add_subplot(339, projection='3d')
    ax.plot_surface(G, T, wT_demand, cmap="viridis")
    ax.set_xlabel("Cost Good")
    ax.set_ylabel("Cost Time")
    ax.set_zlabel("Time Demand T")

    plt.show()

def demand_graphs(): # not updated
    consumer1 = Consumer()

    P, W = np.meshgrid(np.linspace(1, 10, 100), np.linspace(1, 10, 100))

    G = consumer1.good_demand(P, W, constants.interest_rate)
    L = consumer1.leisure_demand(P, W, constants.interest_rate)
    S = consumer1.save_demand(P, W, constants.interest_rate)

    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(P, W, G, cmap="viridis")
    ax.set_xlabel("Price")
    ax.set_ylabel("Wage")
    ax.set_zlabel("Good Demand")

    ax = fig.add_subplot(132, projection='3d')
    ax.plot_surface(P, W, L, cmap="viridis")
    ax.set_xlabel("Price")
    ax.set_ylabel("Wage")
    ax.set_zlabel("Labor Demand")

    ax = fig.add_subplot(133, projection='3d')
    ax.plot_surface(P, W, S, cmap="viridis")
    ax.set_xlabel("Price")
    ax.set_ylabel("Wage")
    ax.set_zlabel("Save Demand")

    plt.show()

def main():
    consumer1 = Consumer()

    lin = np.linspace(1, 20, 100)

    wage = 2

    df = pd.DataFrame(columns=["Price", "Goods", "Labor", "Save", "Leisure", "Interest", "Money"])

    for i in lin:
        # demands = consumer1.optimize(i, wage, constants.interest_rate)
        goods = consumer1.good_demand(i, wage, constants.interest_rate)
        # a_goods = consumer1.additional_good_demand(i, wage, constants.interest_rate)
        # consumer1.purchase(goods, i)
        leisure = consumer1.leisure_demand(wage, i, constants.interest_rate)
        # consumer1.work(labor, wage)
        save = consumer1.save_demand(i, wage, constants.interest_rate)
        # consumer1.purchase(goods, i)
        # consumer1.work(consumer1.max_time - leisure, wage)
        # consumer1.save(save, constants.interest_rate)

        # print(f"Price: {i}, Goods: {demands.x[0]}, Labor: {demands.x[1]}, Save: {demands.x[2]}")
        df.loc[len(df)] = [i, goods, consumer1.max_time - leisure, constants.interest_rate * save, leisure, save, consumer1.money]
        # consumer1.restart()

    print("pass")

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(df["Price"], df["Goods"], color="red", label="Goods")
    ax.plot(df["Price"], df["Labor"], color="blue", label="Labor")
    ax.plot(df["Price"], df["Save"], color="green", label="Save")

    ax.set_xlabel("Price")
    ax.set_ylabel("Quantity")

    ax.legend()

    plt.show()





if __name__ == "__main__":
    main()