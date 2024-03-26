import nevergiveup
import matplotlib.pyplot as plt
import numpy as np

price = 4.646180753165138
wage = 3.200454742912416

A = 5
alpha = 0.75
time = 1
rate = 6

ranges = np.linspace(0.1, 20, 100)
quantity_supply = []
quantity_demand = []
labor_supply = []
labor_demand = []

for i in ranges:

    # given wage
    max_profit, production = nevergiveup.get_max_profit(A, alpha, time, wage, rate, i)

    quantity_supply.append(production)
    quantity_demand.append(nevergiveup.product_demand(i))

    # given price
    max_profit, production = nevergiveup.get_max_profit(A, alpha, time, i, rate, price)


    labor_supply.append(nevergiveup.labor_supply(i))
    labor_demand.append(max_profit.x[0])


max_profit, production = nevergiveup.get_max_profit(A, alpha, time, wage, rate, price)


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