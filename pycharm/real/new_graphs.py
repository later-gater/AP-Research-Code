import firm
import consumer
import constants
import numpy as np
import matplotlib.pyplot as plt

def main():
    price = 112.16794967243362
    wage = 0.20621628879615628
    rate_k = 180.5889264253313

    good_firm = firm.Firm()
    capital_firm = firm.Firm()

    consumer_1 = consumer.Consumer()

    # consumer_1.good_weight = 1.6827961766296413
    # consumer_1.income = 103.43498015278325
    # consumer_1.work_weight = 27.231798498375994

    good_supply = lambda time, A, price, wage, rate_k, rate_z: good_firm.max_profit(time, A, price, wage, rate_k, rate_z)[1]
    capital_supply = lambda time, A, price, wage, rate_k, rate_z: capital_firm.max_profit(time, A, price, wage, rate_k, rate_z)[1]

    labor_supply = lambda price, wage: consumer.Consumer.get_total_work_demand(price, wage)

    good_demand = lambda price, wage: consumer.Consumer.get_total_good_demand(price, wage)

    demands_func = lambda time, A, price, wage, rate_k, rate_z: firm.Firm.get_total_demands(time, A, price, wage, rate_k, rate_z)



    ranger = np.linspace(1.1, 150, 150)

    good_supps = [good_supply(constants.time, constants.A, iter_price, wage, rate_k, constants.rate_z) for iter_price in ranger]
    good_demands = [good_demand(iter_price, wage) for iter_price in ranger]

    capital_supps = [capital_supply(constants.time, constants.A, price, wage, iter_rate_k, constants.rate_z) for iter_rate_k in ranger]
    capital_demands = [demands_func(constants.time, constants.A, price, wage, iter_rate_k, constants.rate_z)[1] for iter_rate_k in ranger]

    labor_supps = [labor_supply(price, iter_wage) for iter_wage in ranger]
    labor_demands = [demands_func(constants.time, constants.A, price, iter_wage, rate_k, constants.rate_z)[0] for iter_wage in ranger]

    fig = plt.figure()

    ax = fig.add_subplot(131)
    ax.plot(ranger, good_supps, color="red", label="Good Supply")
    ax.plot(ranger, good_demands, color="blue", label="Good Demand")
    ax.set_title("Good Supply and Demand")
    ax.set_xlabel("Price")
    ax.set_ylabel("Quantity")
    ax.scatter(price, good_supply(constants.time, constants.A, price, wage, rate_k, constants.rate_z), color="red")
    ax.scatter(price, good_demand(price, wage), color="blue")
    ax.legend()

    ax = fig.add_subplot(132)
    ax.plot(ranger, capital_supps, color="red", label="Capital Supply")
    ax.plot(ranger, capital_demands, color="blue", label="Capital Demand")
    ax.set_title("Capital Supply and Demand")
    ax.set_xlabel("Rate of Capital")
    ax.set_ylabel("Quantity")
    ax.scatter(rate_k, capital_supply(constants.time, constants.A, price, wage, rate_k, constants.rate_z), color="red")
    ax.scatter(rate_k, demands_func(constants.time, constants.A, price, wage, rate_k, constants.rate_z)[1], color="blue")
    ax.legend()

    ax = fig.add_subplot(133)
    ax.plot(ranger, labor_supps, color="red", label="Labor Supply")
    ax.plot(ranger, labor_demands, color="blue", label="Labor Demand")
    ax.set_title("Labor Supply and Demand")
    ax.set_xlabel("Wage")
    ax.set_ylabel("Quantity")
    ax.scatter(wage, labor_supply(price, wage), color="red")
    ax.scatter(wage, demands_func(constants.time, constants.A, price, wage, rate_k, constants.rate_z)[0], color="blue")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()