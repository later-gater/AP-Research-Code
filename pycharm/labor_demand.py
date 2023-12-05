import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

A = 2.0  # constant factor
alpha = 0.75  # labor share
beta = 0.25  # capital share
wage = 1.2
rate = 0.8


def cobb_douglas(labor, capital, alpha, beta, A):
    return A * (labor ** alpha) * (capital ** beta)


def cost_function(labor, capital, wage, rate):
    return (labor*wage) + (capital*rate)


def neg_profit(variables, alpha, beta, A, wage, rate):
    labor, capital = variables
    return -1 * ((A * (labor ** alpha) * (capital ** beta)) - ((labor*wage) + (capital*rate)))


L, K = np.meshgrid(np.linspace(1, 10, 100), np.linspace(1, 10, 100), indexing='ij')

production_values = cobb_douglas(L, K, alpha, beta, A)
cost_values = cost_function(L, K, wage, rate)
profit_values = np.subtract(production_values, cost_values)

max_profit = minimize(neg_profit, [1.0, 1.0], args=(alpha, beta, A, wage, rate), method='L-BFGS-B', bounds=[(0.1, None), (0.1, None)])
fig = plt.figure()

ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot_surface(L, K, production_values, cmap='viridis')
ax.set_xlabel('Labor')
ax.set_ylabel("Capital")
ax.set_zlabel("Production")
ax.set_title("Production Function")

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(L, K, cost_values, cmap='viridis')
ax2.set_xlabel('Labor')
ax2.set_ylabel("Capital")
ax2.set_zlabel("Cost")
ax2.set_title("Cost Function")

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot_surface(L, K, profit_values, cmap='viridis')
ax3.set_xlabel('Labor')
ax3.set_ylabel("Capital")
ax3.set_zlabel("Profit")
ax3.set_title("Profit Function")

plt.show()

pass
