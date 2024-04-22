import firm
import consumer
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence

def draw_graphs(firms: Sequence[firm.Firm], consumers: Sequence[consumer.Consumer], constant: tuple, price: float, wage: float, i_r: float, r_g: float) -> None:
    good_supply = lambda constant, price, wage: sum(
        [iter_firm.max_profit(constant, price, wage, r_g)[1] for iter_firm in firms])
    good_demand = lambda price, wage: sum([iter_consumer.good_demand(price, wage, i_r) for iter_consumer in consumers])

    labor_supply = lambda price, wage: sum([iter_consumer.work_demand(price, wage, i_r) for iter_consumer in consumers])
    labor_demand = lambda constant, price, wage: sum(
        [iter_firm.max_profit(constant, price, wage, r_g)[0].x[0] for iter_firm in firms])

    ranger = np.linspace(0.1, 75, 200)

    good_supps = [good_supply(constant, iter_price, wage) for iter_price in ranger]
    good_demands = [good_demand(iter_price, wage) for iter_price in ranger]

    labor_supps = [labor_supply(price, iter_wage) for iter_wage in ranger]
    labor_demands = [labor_demand(constant, price, iter_wage) for iter_wage in ranger]

    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.plot(ranger, good_supps, color="red", label="Good Supply")
    ax.plot(ranger, good_demands, color="blue", label="Good Demand")
    ax.set_title("Good Supply and Demand")
    ax.set_xlabel("Price")
    ax.set_ylabel("Quantity")
    ax.scatter(price, good_supply(constant, price, wage), color="red")
    ax.scatter(price, good_demand(price, wage), color="blue")
    ax.legend()

    ax = fig.add_subplot(122)
    ax.plot(ranger, labor_supps, color="red", label="Labor Supply")
    ax.plot(ranger, labor_demands, color="blue", label="Labor Demand")
    ax.set_title("Labor Supply and Demand")
    ax.set_xlabel("Wage")
    ax.set_ylabel("Quantity")
    ax.scatter(wage, labor_supply(price, wage), color="red")
    ax.scatter(wage, labor_demand(constant, price, wage), color="blue")
    ax.legend()

    plt.show(block=True)


def main():
    price = 1.2128401876427048
    wage = 9.615067362862249

    good_firm = firm.Firm()
    capital_firm = firm.Firm()

    consumer_1 = consumer.Consumer()

    # consumer_1.good_weight = 1.6827961766296413
    # consumer_1.non_wage_income = 103.43498015278325
    # consumer_1.work_weight = 27.231798498375994

    good_supply = lambda constant, price, wage: good_firm.max_profit(constant, price, wage)[1]

    labor_supply = lambda price, wage: consumer.Consumer.get_total_work_demand(price, wage)

    good_demand = lambda price, wage: consumer.Consumer.get_total_good_demand(price, wage)

    labor_demand = lambda constant, price, wage: firm.Firm.get_total_demands(constant, price, wage)[0]



    ranger = np.linspace(1.1, 75, 150)

    good_supps = [good_supply(constants.constants, iter_price, wage) for iter_price in ranger]
    good_demands = [good_demand(iter_price, wage) for iter_price in ranger]


    labor_supps = [labor_supply(price, iter_wage) for iter_wage in ranger]
    labor_demands = [labor_demand(constants, price, iter_wage) for iter_wage in ranger]

    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.plot(ranger, good_supps, color="red", label="Good Supply")
    ax.plot(ranger, good_demands, color="blue", label="Good Demand")
    ax.set_title("Good Supply and Demand")
    ax.set_xlabel("Price")
    ax.set_ylabel("Quantity")
    ax.scatter(price, good_supply(constants.constants, price, wage), color="red")
    ax.scatter(price, good_demand(price, wage), color="blue")
    ax.legend()


    ax = fig.add_subplot(122)
    ax.plot(ranger, labor_supps, color="red", label="Labor Supply")
    ax.plot(ranger, labor_demands, color="blue", label="Labor Demand")
    ax.set_title("Labor Supply and Demand")
    ax.set_xlabel("Wage")
    ax.set_ylabel("Quantity")
    ax.scatter(wage, labor_supply(price, wage), color="red")
    ax.scatter(wage, labor_demand(constants.constants, price, wage), color="blue")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()