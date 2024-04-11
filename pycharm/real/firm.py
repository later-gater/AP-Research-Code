from typing import Callable

import numpy as np
from scipy.optimize import fsolve, minimize, OptimizeResult, basinhopping
import matplotlib.pyplot as plt
import pandas as pd

class Firm:

	instances = {}

	def __init__(self, alpha_1=0.75, alpha_2=0.75, eos_1=0.75, eos_2=0.75):
		self.money = 20

		self.goods = 0
		self.demands = (0, 0, 0)

		self.alpha_1, self.alpha_2, self.eos_1, self.eos_2 = alpha_1, alpha_2, eos_1, eos_2
		self.constants = (self.alpha_1, self.alpha_2, self.eos_1, self.eos_2)
		Firm.instances[self] = self

	def produce(self, factors: list, wage: float, time: float, A: float, robot_growth: float, rate_k: float, rate_z: float):
		self.goods += self.production_function(factors, time, A, robot_growth)
		self.money -= self.cost_function(factors, wage, rate_k, rate_z)

	def sell(self, quantity: float, price: float):
		self.money += quantity * price
		self.goods -= quantity



	def production_function(self, variables: list,
							time: float, A: float, robot_growth: float) -> float:
		# print(f"time: {time}")
		labor, capital, robots = variables
		alpha_1, alpha_2, eos_1, eos_2 = self.constants
		capital_term = (alpha_1 ** (1 / eos_1)) * (capital ** (((-capital / time) + np.e) * ((eos_1 - 1) / eos_1)))
		labor_inner_term = (alpha_2 ** (1 / eos_2)) * (labor ** (((-labor / time) + np.e) * ((eos_2 - 1) / eos_2)))
		robot_inner_term = ((1 - alpha_2) ** (1 / eos_2)) * (
					robots ** (((-robots / time) + np.e) * ((eos_2 - 1) / eos_2)))
		labor_outer_term = ((1 - alpha_1) ** (1 / eos_1)) * (
					(labor_inner_term + (robot_growth * robot_inner_term)) ** ((eos_2 / (eos_2 - 1)) * (eos_1 - 1) / eos_1))
		final_term = A * ((capital_term + labor_outer_term) ** (eos_1 / (eos_1 - 1)))
		return final_term

	def cost_function(self, variables: list, wage: float, rate_k: float, rate_z: float) -> float:
		labor, capital, robots = variables
		return (wage * labor) + (rate_k * capital) + (rate_z * robots)

	def max_profit(self, constant: tuple[float, float, float, float], price: float, wage: float, robot_growth: float) -> (OptimizeResult, float, float):
		A, time, rate_k, rate_z = constant
		minimized = minimize(lambda x: -1 * ((price * self.production_function(x, time, A, robot_growth)) - self.cost_function(x, wage, rate_k, rate_z)),
							 np.array([1, 1, 1]), bounds=[(0.2, None), (0.2, None), (0.2, None)],
							 constraints={'type': 'ineq', 'fun': lambda x: self.money - self.cost_function(x, wage, rate_k, rate_z)})
		return minimized, self.production_function(minimized.x, time, A, robot_growth), self.cost_function(minimized.x, wage, rate_k, rate_z)

	def equilibrium_cost(self, variables: tuple[float, float], constant: tuple[float, float, float, float],
						 good_demand_func: Callable, labor_supply_func: Callable):
		price, wage = variables


	# def get_optimum_production(self, constant: tuple[float, float, float, float], demand_func: Callable):
	# 	minimized = basinhopping(lambda x: -1 * min(self.max_profit(constant, x[0], x[1]), demand_func(x)) * x[0], [1, 1], )

	@classmethod
	def get_total_production(cls, constant: tuple[float, float, float, float], price: float, wage: float) -> float:
		return sum([cls.instances[instance].max_profit(constant, price, wage)[1] for instance in cls.instances])

	@classmethod
	def get_total_demands(cls, constant: tuple[float, float, float, float], price: float, wage: float) -> (float, float, float):
		demands = [cls.instances[instance].max_profit(constant, price, wage)[0] for instance in cls.instances]
		return sum([demand.x[0] for demand in demands]), sum([demand.x[1] for demand in demands]), sum([demand.x[2] for demand in demands])

def main():
	firm = Firm()
	# max_profit, production = firm.max_profit(1, 5, 5,2.9, 6, 3)
	# print(f"Max Profit: {-max_profit.fun}, Production: {production}, X: {max_profit.x}")
	prices = np.linspace(0.1, 10, 50)
	budgets = np.linspace(15, 95, 5)
	dfs = pd.DataFrame(columns=["Budget", "DF"])
	for budget in budgets:
		firm.money = budget
		df = pd.DataFrame(columns=['Price', 'Max Profit', 'Production', 'Cost', 'Revenue', 'X'])
		for price in prices:
			max_profit, production, cost = firm.max_profit(constants.constants, 20, price)
			print(f"Price: {price}, Max Profit: {-max_profit.fun}, Production: {production}, Cost: {cost}, X: {max_profit.x}")
			df.loc[len(df)] = {'Price': price, 'Max Profit': -max_profit.fun, 'Production': production, 'Cost': cost, 'Revenue': price*production, 'X': max_profit.x}
		dfs.loc[len(dfs)] = {"Budget": budget, "DF": df}

	fig = plt.figure()

	# ax = fig.add_subplot(111)
	# ax.plot(df['Price'], df['Production'], color="red", label="Production")
	# ax.plot(df['Price'], df['Max Profit'], color="blue", label="Max Profit")
	# ax.plot(df['Price'], df['Revenue'], color="yellow", label="Revenue")
	# ax.plot(df['Price'], df['Cost'], color="green", label="Cost")
	# ax.set_xlabel('Price')
	# ax.set_ylabel('Production <> Max Profit <> Cost')
	# ax.legend()

	ax = fig.add_subplot(141)
	ax.plot(dfs.iloc[0,1]["Price"], dfs.iloc[0,1]["Production"], color="red", label=dfs.iloc[0,0])
	ax.plot(dfs.iloc[1,1]["Price"], dfs.iloc[1,1]["Production"], color="blue", label=dfs.iloc[1,0])
	ax.plot(dfs.iloc[2,1]["Price"], dfs.iloc[2,1]["Production"], color="green", label=dfs.iloc[2,0])
	ax.plot(dfs.iloc[3,1]["Price"], dfs.iloc[3,1]["Production"], color="yellow", label=dfs.iloc[3,0])
	ax.plot(dfs.iloc[4,1]["Price"], dfs.iloc[4,1]["Production"], color="purple", label=dfs.iloc[4,0])
	ax.set_xlabel('Price')
	ax.set_ylabel('Production')
	ax.legend()

	ax = fig.add_subplot(142)
	ax.plot(dfs.iloc[0,1]["Price"], dfs.iloc[0,1]["Max Profit"], color="red", label=dfs.iloc[0,0])
	ax.plot(dfs.iloc[1,1]["Price"], dfs.iloc[1,1]["Max Profit"], color="blue", label=dfs.iloc[1,0])
	ax.plot(dfs.iloc[2,1]["Price"], dfs.iloc[2,1]["Max Profit"], color="green", label=dfs.iloc[2,0])
	ax.plot(dfs.iloc[3,1]["Price"], dfs.iloc[3,1]["Max Profit"], color="yellow", label=dfs.iloc[3,0])
	ax.plot(dfs.iloc[4,1]["Price"], dfs.iloc[4,1]["Max Profit"], color="purple", label=dfs.iloc[4,0])
	ax.set_xlabel('Price')
	ax.set_ylabel('Max Profit')
	ax.legend()

	ax = fig.add_subplot(143)
	ax.plot(dfs.iloc[0,1]["Price"], dfs.iloc[0,1]["Revenue"], color="red", label=dfs.iloc[0,0])
	ax.plot(dfs.iloc[1,1]["Price"], dfs.iloc[1,1]["Revenue"], color="blue", label=dfs.iloc[1,0])
	ax.plot(dfs.iloc[2,1]["Price"], dfs.iloc[2,1]["Revenue"], color="green", label=dfs.iloc[2,0])
	ax.plot(dfs.iloc[3,1]["Price"], dfs.iloc[3,1]["Revenue"], color="yellow", label=dfs.iloc[3,0])
	ax.plot(dfs.iloc[4,1]["Price"], dfs.iloc[4,1]["Revenue"], color="purple", label=dfs.iloc[4,0])
	ax.set_xlabel('Price')
	ax.set_ylabel('Revenue')
	ax.legend()

	ax = fig.add_subplot(144)
	ax.plot(dfs.iloc[0,1]["Price"], dfs.iloc[0,1]["Cost"], color="red", label=dfs.iloc[0,0])
	ax.plot(dfs.iloc[1,1]["Price"], dfs.iloc[1,1]["Cost"], color="blue", label=dfs.iloc[1,0])
	ax.plot(dfs.iloc[2,1]["Price"], dfs.iloc[2,1]["Cost"], color="green", label=dfs.iloc[2,0])
	ax.plot(dfs.iloc[3,1]["Price"], dfs.iloc[3,1]["Cost"], color="yellow", label=dfs.iloc[3,0])
	ax.plot(dfs.iloc[4,1]["Price"], dfs.iloc[4,1]["Cost"], color="purple", label=dfs.iloc[4,0])
	ax.set_xlabel('Price')
	ax.set_ylabel('Cost')
	ax.legend()



	plt.show()
	pass
	print("pass")

if __name__ == "__main__":
	main()