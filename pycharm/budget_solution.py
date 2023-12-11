import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import fsolve

class Firm:
    def __init__(self, alpha, beta, start_money):
        self.alpha = alpha
        self.beta = beta
        self.money = start_money
        self.price = 2  # create price market
        self.raw_mat_cost = 1  # do i need market for this?
        self.profit_per_unit = self.price - self.raw_mat_cost

    def get_labor_demand(self, wage):
        return self.money / (wage * (1 + (self.beta / self.alpha)))


    def get_capital_demand(self, rate):
        return self.money / (rate * (1 + (self.alpha / self.beta)))


def get_labor_supply(wage):
    return 2 * wage


def find_equilibrium_labor(wage, arg):
    firm = arg[0]
    return firm.get_labor_demand(wage) - get_labor_supply(wage)


def get_capital_supply(rate):
    return 1.5 * rate


def find_equilibrium_capital(rate, arg):
    firm = arg[0]
    return firm.get_capital_demand(rate) - get_capital_supply(rate)


def main():
    firm = Firm(0.75, 0.25, 1000)

    equilibrium_wage = fsolve(find_equilibrium_labor, np.array([1]), args=([firm]))
    equilibrium_rate = fsolve(find_equilibrium_capital, np.array([1]), args=([firm]))

    wages = np.linspace(5, 30, 251)
    rates = np.linspace(5, 30, 251)
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(firm.get_labor_demand(wages), wages, color='red')
    ax.plot(get_labor_supply(wages), wages, color='blue')
    ax.plot(find_equilibrium_labor(wages, [firm]), wages, color='green')
    ax.plot(np.linspace(0, 0, 251), wages, color='black')
    ax.scatter(get_labor_supply(equilibrium_wage[0]), equilibrium_wage[0])
    ax.set_xlabel("Quantity of Labor")
    ax.set_ylabel("Wages")

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(firm.get_capital_demand(rates), rates, color='red')
    ax.plot(get_capital_supply(rates), rates, color='blue')
    ax.plot(find_equilibrium_capital(rates, [firm]), rates, color='green')
    ax.plot(np.linspace(0, 0, 251), rates, color='black')
    ax.scatter(get_capital_supply(equilibrium_rate[0]), equilibrium_rate[0])
    ax.set_xlabel("Quantity of Capital")
    ax.set_ylabel("Rates")

    plt.show()

    pass


if __name__ == "__main__":
    main()
