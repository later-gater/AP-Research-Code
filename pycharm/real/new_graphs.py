import firm
import consumer
import constants
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence

def draw_graphs(firms: Sequence[firm.Firm], consumers: Sequence[consumer.Consumer], price: float, wage: float) -> None:
    good_supply = lambda time, A, price, wage, rate_k, rate_z: sum(
        [iter_firm.max_profit(time, A, price, wage, rate_k, rate_z)[1] for iter_firm in firms])
    good_demand = lambda price, wage: sum([iter_consumer.good_demand(price, wage) for iter_consumer in consumers])

    labor_supply = lambda price, wage: sum([iter_consumer.work_demand(price, wage) for iter_consumer in consumers])
    labor_demand = lambda time, A, price, wage, rate_k, rate_z: sum(
        [iter_firm.max_profit(time, A, price, wage, rate_k, rate_z)[0].x[0] for iter_firm in firms])

    ranger = np.linspace(1.1, 75, 150)

    good_supps = [good_supply(constants.time, constants.A, iter_price, wage, constants.rate_k, constants.rate_z) for iter_price in ranger]
    good_demands = [good_demand(iter_price, wage) for iter_price in ranger]

    labor_supps = [labor_supply(price, iter_wage) for iter_wage in ranger]
    labor_demands = [labor_demand(constants.time, constants.A, price, iter_wage, constants.rate_k, constants.rate_z) for iter_wage in ranger]

    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.plot(ranger, good_supps, color="red", label="Good Supply")
    ax.plot(ranger, good_demands, color="blue", label="Good Demand")
    ax.set_title("Good Supply and Demand")
    ax.set_xlabel("Price")
    ax.set_ylabel("Quantity")
    ax.scatter(price, good_supply(constants.time, constants.A, price, wage, constants.rate_k, constants.rate_z), color="red")
    ax.scatter(price, good_demand(price, wage), color="blue")
    ax.legend()

    ax = fig.add_subplot(122)
    ax.plot(ranger, labor_supps, color="red", label="Labor Supply")
    ax.plot(ranger, labor_demands, color="blue", label="Labor Demand")
    ax.set_title("Labor Supply and Demand")
    ax.set_xlabel("Wage")
    ax.set_ylabel("Quantity")
    ax.scatter(wage, labor_supply(price, wage), color="red")
    ax.scatter(wage, labor_demand(constants.time, constants.A, price, wage, constants.rate_k, constants.rate_z), color="blue")
    ax.legend()

    plt.show()

def main():
    price = 1.2128401876427048
    wage = 9.615067362862249

    good_firm = firm.Firm()
    capital_firm = firm.Firm()

    consumer_1 = consumer.Consumer()

    # consumer_1.good_weight = 1.6827961766296413
    # consumer_1.income = 103.43498015278325
    # consumer_1.work_weight = 27.231798498375994

    good_supply = lambda time, A, price, wage, rate_k, rate_z: good_firm.max_profit(time, A, price, wage, rate_k, rate_z)[1]

    labor_supply = lambda price, wage: consumer.Consumer.get_total_work_demand(price, wage)

    good_demand = lambda price, wage: consumer.Consumer.get_total_good_demand(price, wage)

    labor_demand = lambda time, A, price, wage, rate_k, rate_z: firm.Firm.get_total_demands(time, A, price, wage, rate_k, rate_z)[0]



    ranger = np.linspace(1.1, 75, 150)

    good_supps = [good_supply(constants.time, constants.A, iter_price, wage, constants.rate_k, constants.rate_z) for iter_price in ranger]
    good_demands = [good_demand(iter_price, wage) for iter_price in ranger]


    labor_supps = [labor_supply(price, iter_wage) for iter_wage in ranger]
    labor_demands = [labor_demand(constants.time, constants.A, price, iter_wage, constants.rate_k, constants.rate_z) for iter_wage in ranger]

    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.plot(ranger, good_supps, color="red", label="Good Supply")
    ax.plot(ranger, good_demands, color="blue", label="Good Demand")
    ax.set_title("Good Supply and Demand")
    ax.set_xlabel("Price")
    ax.set_ylabel("Quantity")
    ax.scatter(price, good_supply(constants.time, constants.A, price, wage, constants.rate_k, constants.rate_z), color="red")
    ax.scatter(price, good_demand(price, wage), color="blue")
    ax.legend()


    ax = fig.add_subplot(122)
    ax.plot(ranger, labor_supps, color="red", label="Labor Supply")
    ax.plot(ranger, labor_demands, color="blue", label="Labor Demand")
    ax.set_title("Labor Supply and Demand")
    ax.set_xlabel("Wage")
    ax.set_ylabel("Quantity")
    ax.scatter(wage, labor_supply(price, wage), color="red")
    ax.scatter(wage, labor_demand(constants.time, constants.A, price, wage, constants.rate_k, constants.rate_z), color="blue")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()