import nevergiveup
import matplotlib.pyplot as plt
import numpy as np

price = 9.6
wage = 2.9

A = 5
alpha_1 = 0.75
alpha_2 = 0.75
eos_1 = 0.75
eos_2 = 0.75
constants = (alpha_1, alpha_2, eos_1, eos_2)
time = 1
rate_k = 6
rate_z = 3

ranges = np.linspace(0.1, 20, 100)
quantity_supply = []
quantity_demand = []
labor_supply = []
labor_demand = []

for i in ranges:

    # given wage
    max_profit, production = nevergiveup.get_max_profit(constants, time, A, wage, rate_k, rate_z, i)

    quantity_supply.append(production)
    quantity_demand.append(nevergiveup.product_demand(i))

    # given price
    max_profit, production = nevergiveup.get_max_profit(constants, time, A, i, rate_k, rate_z, price)


    labor_supply.append(nevergiveup.labor_supply(i))
    labor_demand.append(max_profit.x[0])


max_profit, production = nevergiveup.get_max_profit(constants, time, A, wage, rate_k, rate_z, price)


fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(ranges, quantity_supply, color="red", label="Quantity Supplied")
ax.plot(ranges, quantity_demand, color="blue", label="Quantity Demanded")
ax.scatter(price, production, color="red")
ax.set_xlabel(f"Price (given wage = {wage}")
ax.set_ylabel("Quantity")

ax = fig.add_subplot(122)
ax.plot(ranges, labor_supply, color="red", label="Labor Supplied")
ax.plot(ranges, labor_demand, color="blue", label="Labor Demanded")
ax.scatter(wage, max_profit.x[0], color="red")
ax.set_xlabel(f"Wage (given price = {price})")
ax.set_ylabel("Labor")


plt.show()