import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import time

wage = 1.2
rate = 0.8

class Firm:
    def __init__(self, A: float, alpha: float, beta: float):
        self.A = A
        self.alpha = alpha
        self.beta = beta
        self.rts = self.alpha + self.beta  # stands for return to scale. if rts<1 -> decreasing returns to scale. if rts>1 -> increasing returns to scale
        # if rts=1 -> constant returns to scale -> perfectly elastic supply

    def find_max_output(self, price, variables: list[wage: float, rate: float]) -> float:
        """
        Given wage, rate, and price, returns the optimal output as defined by the minimized cost function:\n
        C = wL + rK
        (w.r.t) Y = A * (L**alpha) * (K**beta)
        """
        wage, rate = variables
        c_term = 1 / self.rts
        a_term = self.A ** (-1 / self.rts)
        w_term = wage ** (self.alpha / self.rts)  # double check
        r_term = rate ** (self.beta / self.rts)  # double check
        alpha_beta_term = ((self.alpha / self.beta) ** (self.beta / self.rts)) + ((self.beta / self.alpha) ** (self.alpha / self.rts))  # double check
        main_term = price * ((c_term * a_term * w_term * r_term * alpha_beta_term) ** -1)  # double check
        max_output = main_term ** (self.rts / (1 - self.rts))
        return max_output

    def find_optimums(self):
        """
        First solves
        :return:
        """
        # optimal_output = fsolve(lambda price: )
        pass

def labor_supply(wage):
    return wage**1.2

def good_demand(price):
    return 10 - (2*price)



def main():
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)

    firm = Firm(2, 0.8, 0.4)

    prices = np.linspace(0.5, 4, 36)

    supply = firm.find_max_output(prices, [wage, rate])

    demand = good_demand(prices)

    ax.plot(prices, supply, color="red")
    ax.plot(prices, demand, color="blue")
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Price")

    plt.show()

    pass



if __name__ == "__main__":
    main()


