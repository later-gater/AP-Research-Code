import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

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

W, R = np.meshgrid(np.linspace(0.1, 5, 50), np.linspace(0.1, 5, 50), indexing='ij')

production_values = cobb_douglas(L, K, alpha, beta, A)
cost_values = cost_function(L, K, wage, rate)
profit_values = np.subtract(production_values, cost_values)

start_time = time.time()
max_profit_funcs = np.vectorize(lambda w, r: minimize(neg_profit, np.array([1.0, 1.0]), args=(alpha, beta, A, w, r), method='L-BFGS-B', bounds=[(0.1, None), (0.1, None)])).__call__(W, R)

max_profit_values = np.full_like(W, fill_value=None, dtype=np.float32)

mapped_profits = np.full_like(W, fill_value=None, dtype=np.dtype)

successes = 0
for i, row in enumerate(max_profit_funcs):
    for j, cell in enumerate(row):
        if cell.success:
            max_profit_values[i, j] = -cell.fun
            mapped_profits[i, j] = (W[i, j], R[i, j], max_profit_values[i, j])
            successes += 1
        else:
            mapped_profits[i, j] = (W[i, j], R[i, j], None)
end_time = time.time()

total_time = end_time-start_time

fig = plt.figure()
#
# ax = fig.add_subplot(2, 2, 1, projection='3d')
# ax.plot_surface(L, K, production_values, cmap='viridis')
# ax.set_xlabel('Labor')
# ax.set_ylabel("Capital")
# ax.set_zlabel("Production")
# ax.set_title("Production Function")
#
# ax2 = fig.add_subplot(2, 2, 2, projection='3d')
# ax2.plot_surface(L, K, cost_values, cmap='viridis')
# ax2.set_xlabel('Labor')
# ax2.set_ylabel("Capital")
# ax2.set_zlabel("Cost")
# ax2.set_title("Cost Function")
#
# ax3 = fig.add_subplot(2, 2, 3, projection='3d')
# ax3.plot_surface(L, K, profit_values, cmap='viridis')
# ax3.set_xlabel('Labor')
# ax3.set_ylabel("Capital")
# ax3.set_zlabel("Profit")
# ax3.set_title("Profit Function")
#
ax4 = fig.add_subplot(1, 1, 1, projection='3d')
ax4.plot_surface(W, R, max_profit_values, cmap='viridis')
ax4.set_xlabel('Wage')
ax4.set_ylabel("Rate")
ax4.set_zlabel("Profit")
ax4.set_title("Profit by wage and rate")

plt.show()


time.sleep(10)

