import firm
import consumer
import equilibrium
from typing import Sequence
import pandas as pd
import numpy as np
import dill


class Market:
    def __init__(self, firms: Sequence[firm.Firm], consumers: Sequence[consumer.Consumer], constants: dict):
        self.firms = firms
        self.consumers = consumers
        self.constants = constants # A, time, rate_k, rate_z, interest_rate, robot_growth_func
        self.trash_money, self.price, self.wage, self.error = 0, 0, 0, 0

    def get_equilibrium(self) -> (float, float):
        equi = equilibrium.find_equilibrium(self.firms, self.consumers, (self.constants["A"], self.constants["time"], self.constants["rate_k"], self.constants["rate_z"], self.constants["interest_rate"], self.constants["robot_growth_func"](self.constants["time"])), 20)
        # new_graphs.draw_graphs(self.firms, self.consumers, (self.constants["A"], self.constants["time"], self.constants["rate_k"], self.constants["rate_z"]), equi.x[0], equi.x[1], self.constants["interest_rate"], self.constants["robot_growth_func"](self.constants["time"])
        return equi.x[0], equi.x[1], equi.fun

    def get_money_in_circulation(self, trash=False) -> float:
        return sum([f.money for f in self.firms]) + sum([c.money for c in self.consumers]) + sum([c.money_saved for c in self.consumers]) + (self.trash_money*int(trash))

    def set_market(self):
        self.price, self.wage, self.error = self.get_equilibrium()
        if self.error > 1:
            print("Could not converge: cost is ", self.error)
        for f in self.firms:
            mini, _, _ = f.max_profit((self.constants["A"], self.constants["time"], self.constants["rate_k"],
                                       self.constants["rate_z"]), self.price, self.wage, self.constants["robot_growth_func"](self.constants["time"]))
            f.demands = mini.x
        for c in self.consumers:
            c.demands = c.get_demands(self.price, self.wage, self.constants["interest_rate"])

    def exchange_goods(self, c: consumer.Consumer, f: firm.Firm):
        goods_exchanged = np.fmin(c.max_purchase(self.price), np.fmin(c.demands[0], f.goods))
        print(f"{c.name} bought {goods_exchanged} goods")
        f.sell(goods_exchanged, self.price)
        c.purchase(goods_exchanged, self.price)
        print(f"firm has {f.goods} goods left")


    def resolve_market(self):
        for f in self.firms:
            print(f"goods before: {f.goods}")
            f.produce([np.fmin(sum([c.demands[1] for c in self.consumers]), f.demands[0]), f.demands[1], f.demands[2]],
                      self.wage, self.constants["time"], self.constants["A"], self.constants["robot_growth_func"](self.constants["time"]),
                      self.constants["rate_k"], self.constants["rate_z"])
            print(f"goods after: {f.goods}")
        for c in self.consumers:
            c.work(np.fmin(sum([f.demands[0] for f in self.firms]), c.demands[1]), self.wage)
            print(f"{c.name} worked {np.fmin(sum([f.demands[0] for f in self.firms]), c.demands[1])} hrs")
            for f in self.firms:
                self.exchange_goods(c, f)
            c.save(c.money)
            c.set_util()

    def run_market(self, output: pd.DataFrame):
        self.set_market()
        self.resolve_market()
        data = {
                "Convergence Error": self.error,
                "Money in Circulation": self.get_money_in_circulation(),
                "Firm Money": sum([f.money for f in self.firms]),
                "Leftover Goods": sum([f.goods for f in self.firms]),
                "Firm Labor Demand": sum([f.demands[0] for f in self.firms]),
                "Firm Capital Demand": sum([f.demands[1] for f in self.firms]),
                "Firm Robot Demand": sum([f.demands[2] for f in self.firms]),
                "Robot Growth": float(self.constants["robot_growth_func"](self.constants["time"])),
                "Price": self.price,
                "Wage": self.wage,
                "Consumer Start Moneys": pd.Series({c.name: c.starting_money for c in self.consumers}),
                "Utils": pd.Series({c.name: c.util for c in self.consumers}),
                "Times Worked": pd.Series({c.name: c.time_working for c in self.consumers}),
                "Leisure": pd.Series({c.name: c.leisure for c in self.consumers}),
                "Money Earned": pd.Series({c.name: c.time_working * self.wage for c in self.consumers}),
                "Goods Purchased": pd.Series({c.name: c.num_goods for c in self.consumers}),
                "Money Spent": pd.Series({c.name: c.num_goods * self.price for c in self.consumers}),
                "Money Saved": pd.Series({c.name: c.money_saved for c in self.consumers}),
                "Interest Earned": pd.Series({c.name: c.money_saved * self.constants["interest_rate"] for c in self.consumers})}
        data = pd.DataFrame(data)
        data = data.set_index(pd.MultiIndex.from_product([[self.constants["time"]], data.index]))
        for c in self.consumers:
            c.next_year(self.constants["interest_rate"])
        self.constants["time"] = self.constants["time"] + 1
        return pd.concat([output, pd.DataFrame(data)])


def run_market(index, file_name, alpha_2, eos_1, eos_2, robot_growth_func, rate_k, rate_z, incomes):
    print(f"Running Market {file_name}")
    var = {
            "index": index, "file_name": file_name, "alpha_1": 0.35, "alpha_2": alpha_2, "eos_1": eos_1,
            "eos_2": eos_2, "robot_growth_func": robot_growth_func, "A": 20, "time": 1, "rate_k": rate_k,
            "rate_z": rate_z, "interest_rate": 0.02, "incomes": incomes, "p_leisure": 0.425, "p_good": 0.425, "p_save": 0.05
          }

    good_firm = firm.Firm(var["alpha_1"], var["alpha_2"], var["eos_1"], var["eos_2"])

    upper = consumer.Consumer("upper", var["incomes"][0], var["p_leisure"], var["p_good"], var["p_save"])
    middle = consumer.Consumer("middle", var["incomes"][1], var["p_leisure"], var["p_good"], var["p_save"])
    lower = consumer.Consumer("lower", var["incomes"][2], var["p_leisure"], var["p_good"], var["p_save"])

    market = Market([good_firm], [upper, middle, lower],
                    {x:var[x] for x in ["A", "time", "rate_k", "rate_z", "interest_rate", "robot_growth_func"]})

    data = pd.DataFrame(columns=["Convergence Error", "Money in Circulation", "Firm Money", "Leftover Goods",
                                 "Firm Labor Demand", "Firm Capital Demand", "Firm Robot Demand", "Robot Growth",
                                 "Price", "Wage", "Consumer Start Moneys", "Utils", "Times Worked", "Leisure",
                                 "Money Earned", "Goods Purchased", "Money Spent", "Money Saved", "Interest Earned"])
    for i in range(8):
        data = market.run_market(data)
        print(f"year {i + 1} completed")
    var["robot_growth_func"] = dill.dumps(var["robot_growth_func"])
    data.attrs = var
    return data

if __name__ == "__main__":
    main()