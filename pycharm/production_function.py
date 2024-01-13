import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
import pandas as pd
import time

def production_function(variables, A, alpha, time, price):

    labor, capital = variables
    def phi(x, t):
        return (1/np.exp(1)) + np.exp((-x / np.exp(t)))

    labor_term = (alpha*labor)**phi(labor, time)
    capital_term = ((1-alpha)*capital)**phi(capital, time)

    return price * A * labor_term * capital_term
    # return labor_term  # debugging

def cost_function(variables, A, alpha, time, wage, rate, price):
    labor, capital = variables
    return (wage*labor) + (rate*capital)


def neg_profit_function(variables, A, alpha, time, wage, rate, price):
    labor, capital = variables
    return -1 * ((price * production_function(variables, A, alpha, time, price)) - (wage * labor + rate * capital))

def production_gradient(variables, A, alpha, time, wage, rate, price):
    labor, capital = variables

    def phi(x, t):  # since i use this twice, should probably define outside of local scope TODO
        return (1/np.exp(1)) + np.exp((-x / np.exp(t)))


    labor_term = labor ** (phi(labor, time) - 1)
    alpha_term_labor = alpha ** phi(labor, time)
    phi_prime_term_labor = -1 * np.exp((-labor / np.exp(time)) - 1)
    misc_term_labor = (labor*np.log(labor)) + (labor*np.log(alpha)) - (np.exp(labor/np.exp(time))) - np.exp(1)
    derived_labor_part = labor_term * alpha_term_labor * phi_prime_term_labor * misc_term_labor

    capital_part = ((1 - alpha) * capital) ** phi(capital, time)
    dL = -1 * ((A * price * derived_labor_part * capital_part) - wage)

    capital_term = capital ** (phi(capital, time) - 1)
    alpha_term_capital = (1-alpha) ** phi(capital, time)
    phi_prime_term_capital = -1 * np.exp((-capital / np.exp(time)) - 1)
    misc_term_capital = (capital*np.log(capital)) + (capital*np.log(1-alpha)) - (np.exp(capital/np.exp(time))) - np.exp(1)
    derived_capital_part = capital_term * alpha_term_capital * phi_prime_term_capital * misc_term_capital

    labor_part = (alpha*labor) ** phi(labor, time)
    dK = -1 * ((A * price * labor_part * derived_capital_part) - rate)

    return np.array([dL, dK])



def main():
    A = 5
    alpha = 0.75
    time = 1

    W = np.linspace(0.1, 10, 100)
    R = np.linspace(0.1, 10, 100)
    P = np.linspace(0.1, 10, 100)

    mapped_roots_fsolve = []
    roots_fsolve = np.full_like([[P for r in R] for w in W], fill_value=None, dtype=np.dtype)

    pass

    for i, wage in enumerate(W):
        print(f'w: {wage}')
        for j, rate in enumerate(R):
            for k, price in enumerate(P):

                roots_fsolve[i, j, k] = fsolve(production_gradient, x0=np.array([100.,100.]), args=(A, alpha, time, wage, rate, price))
                labor_fsolve = roots_fsolve[i][j][k][0]
                capital_fsolve = roots_fsolve[i][j][k][1]
                R_fsolve = production_function([labor_fsolve, capital_fsolve], A, alpha, time, price)
                Y_fsolve = R_fsolve/price
                LC_fsolve = labor_fsolve * wage
                KC_fsolve = capital_fsolve * rate
                TC_fsolve = LC_fsolve + KC_fsolve
                AC_fsolve = TC_fsolve / Y_fsolve
                P_fsolve = R_fsolve - TC_fsolve
                mapped_roots_fsolve.append({
                    "w": wage,
                    "r": rate,
                    "p": price,
                    "L": labor_fsolve,
                    "K": capital_fsolve,
                    "Y": Y_fsolve,
                    "R": R_fsolve,
                    "LC": LC_fsolve,
                    "KC": KC_fsolve,
                    "TC": TC_fsolve,
                    "P": P_fsolve,
                    "AC": AC_fsolve
                })
                pass


    gradient_roots = pd.DataFrame(mapped_roots_fsolve)


    fig = plt.figure()

    # ax = fig.add_subplot(2, 2, 1, projection='3d')
    # ax.plot_surface(L, K, productions, cmap="viridis")
    # ax.set_xlabel("Labor")
    # ax.set_ylabel("Capital")
    # ax.set_zlabel("Production")
    #
    # ax = fig.add_subplot(2, 2, 3, projection='3d')
    # ax.plot_surface(L, K, gradients[0], cmap="viridis")
    # ax.set_xlabel("Labor")
    # ax.set_ylabel("Capital")
    # ax.set_zlabel("dL")
    #
    # ax = fig.add_subplot(2, 2, 4, projection='3d')
    # ax.plot_surface(L, K, gradients[1], cmap="viridis")
    # ax.set_xlabel("Labor")
    # ax.set_ylabel("Capital")
    # ax.set_zlabel("dK")



    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(L, productions, color="red")
    # ax.plot(L, gradients, color='blue')
    # ax.set_xlabel("Labor")


    # plt.show()
    pass

if __name__ == "__main__":
    main()
