import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

x = np.linspace(1, 21, 200)

# plt.plot(x, (x), color='red')
#
# plt.show()


capital_cost = 4
# real_interest_rate = 0.02 # later


class Firm:
    def __init__(self):
        self.labor = 1
        self.capital = 1
        self.alpha = 0.75
        self.beta = 0.25
        self.mult = 20

        self.labor_demand = 0
        self.money = 100

    def produce(self):
        self.money += self.mult * (self.labor ** self.alpha) * (self.capital ** self.beta)

    def find_mpl(self, labor):
        return self.mult*self.alpha*(labor**(self.alpha-1))

    def profits_from_marginal_labor(self, labor, wage):
        return self.find_mpl(labor) - wage

    def find_labor_demand(self, wage):
        try:
            self.labor_demand = fsolve(self.profits_from_marginal_labor, 1, args=wage)
            if self.labor_demand == 1.:
                self.labor_demand = [0]
        except RuntimeWarning:
            self.labor_demand = [0]
        return self.labor_demand


def labor_supply(wage):
    return 10 * wage


def main():
    firm = Firm()
    # plt.plot(x, firm.find_mpl(x), color='red')
    # plt.plot(x, [wage for i in x], color='black')
    # plt.plot(x, firm.profits_from_marginal_labor(x), color='blue')
    # print(firm.find_profits_x_int())
    y = []
    for i in x:
        y.append(firm.find_labor_demand(i))

    plt.plot(x, y, color="red")
    plt.plot(x, labor_supply(x), color='blue')

    plt.show()
    pass


if __name__ == "__main__":
    main()
