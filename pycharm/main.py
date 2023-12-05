import pandas as pd
import random
import numpy as np

import matplotlib.pyplot as plt


class Good:
    def __init__(self, name, marg_util_func, pref, init_price):
        self.name = name
        self.marg_util_func = marg_util_func
        self.pref = pref
        self.df = pd.DataFrame()
        self.price = init_price

    def marginal_util(self, num_owned):
        return self.marg_util_func(num_owned)

    def produce(self, producer):
        pass

    def gen_pref(self):
        return random.betavariate(self.pref[0], self.pref[1])


def init(num_pop, num_firm, num_goods):
    goods = [
        Good("Clothes", lambda num_owned: (2 ** (num_owned * -1)) * 10, (4, 8), 100),
        Good("Food", lambda num_owned: (2**(num_owned * -1)) * 10, (8, 2), 20)
    ]
    pop = pd.DataFrame()
    pop["money"] = [random.randint(1000, 2000) for i in range(num_pop)]
    pop["util_weights"] = pop.apply(lambda row: [good.gen_pref() for good in goods], axis=1)
    pop["goods_owned"] = [[0 for j in range(num_goods)] for i in range(num_pop)]

    firm = pd.DataFrame()

    return pop, firm, goods


def gen_marg_util(pop, goods):
    pop["marginal_utils"] = pop.apply(
        lambda row: [good.marginal_util(row["goods_owned"][i]) * row["util_weights"][i] for i, good in
                     enumerate(goods)],
        axis=1
    )
    good_prices = [good.price for good in goods]
    pop["marginal_util/dollar"] = pop.apply(
        lambda row: np.divide(row["marginal_utils"], good_prices), axis=1
    )
    pass


def main():
    pop, firm, goods = init(100000, 10, 2)
    gen_marg_util(pop, goods)

    plt.figure(figsize=(8, 4))
    plt.hist([person[1] for person in pop["util_weights"]], bins=100)
    plt.show()
    pass


if __name__ == '__main__':
    main()
