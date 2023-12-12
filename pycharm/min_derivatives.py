import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import time

A = 2.0  # constant factor
alpha = 0.75  # labor share
beta = 0.25  # capital share
wage = 2.6
rate = 0.1


def cobb_douglas(labor, capital, alpha, beta, A):
    return A * (labor ** alpha) * (capital ** beta)


def cost_function(labor, capital, wage, rate):
    return (labor*wage) + (capital*rate)


def profit(variables, alpha, beta, A, wage, rate):
    labor, capital = variables
    return (A * (labor ** alpha) * (capital ** beta)) - ((labor*wage) + (capital*rate))


def profit_gradient(variables, alpha, beta, A, wage, rate):
    labor, capital = variables
    dL = A * alpha * (labor**(alpha-1)) * (capital**beta) - wage
    dK = A * beta * (capital**(beta-1)) * (labor**alpha) - rate
    return np.array([dL, dK])



# W, R = np.meshgrid(np.linspace(1, 2, 11), np.linspace(1, 2, 11), indexing='ij')
#
#
# mapped_roots = []
# roots = np.full_like(W, fill_value=None, dtype=np.dtype)
#
# for i, w in enumerate(W):
#     for j, r in enumerate(R):
#         roots[i, j] = fsolve(profit_gradient, x0=np.array([1.,1.]), args=(alpha, beta, A, w[j], r[j]))
#         mapped_roots.append({
#             "W": w[j],
#             "R": r[j],
#             "root": roots[i][j]
#         })
#
# labor_roots = np.vectorize(lambda zero: zero[0])(roots)
# capital_roots = np.vectorize(lambda zero: zero[1])(roots)
#
# fig = plt.figure()
#
# ax = fig.add_subplot(1, 2, 1, projection="3d")
# ax.plot_surface(W, R, labor_roots, cmap='viridis')
# ax.set_xlabel("Wage")
# ax.set_ylabel("Rate")
# ax.set_zlabel("labor")
#
# ax = fig.add_subplot(1, 2, 2, projection="3d")
# ax.plot_surface(W, R, capital_roots, cmap='viridis')
# ax.set_xlabel("Wage")
# ax.set_ylabel("Rate")
# ax.set_zlabel("capital")
#
# plt.show()

root = fsolve(profit_gradient, x0=np.array([1.,1.]), args=(alpha, beta, A, 1.2, 1.9), xtol=1000, maxfev=5000)

fig = plt.figure()

L, K = np.meshgrid(np.linspace(root[0]-300, root[0]+300, 501), np.linspace(root[1]-300, root[1]+300, 501), indexing='ij')

gradient_values = profit_gradient([L, K], alpha, beta, A, 1.2, 1.9)
# labor_gradient =
# capital_gradient = np.vectorize(lambda gradient: gradient[1])(gradient_values)


ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot_surface(L, K, gradient_values[0], cmap='viridis')
ax.set_xlabel("Labor")
ax.set_ylabel("Capital")
ax.set_zlabel("Gradient")
ax.scatter(root[0], root[1], profit_gradient(root, alpha, beta, A, 1.2, 1.9)[0], color='red')
ax.scatter(root[0], root[1], 0, color='green')
ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.plot_surface(L, K, gradient_values[1], cmap='viridis')
ax.set_xlabel("Labor")
ax.set_ylabel("Capital")
ax.set_zlabel("Gradient")
ax.scatter(root[0], root[1], profit_gradient(root, alpha, beta, A, 1.2, 1.9)[1], color='red')
ax.scatter(root[0], root[1], 0, color='green')

#
plt.show()

pass




