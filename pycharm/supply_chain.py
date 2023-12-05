import pandas as pd
import random


class HirePool(object):
    employees = []
    employers = []

    def foo(self):
        pass


class Firm:
    def __init__(self):
        self.workers = []

    def send_hire_request(self, skill, quantity, min_pay, max_pay):
        pass  # vectorize some stuff with dataframe i think


class HireRequest:
    def __init__(self, skill, min_pay, max_pay):
        self.skill = skill
        self.min_pay = min_pay
        self.max_pay = max_pay

    def complete(self):
        del self



class AlreadyEmployed(Exception):
    pass


class Worker:
    def __init__(self, number):
        self.number = number
        self.employed = False
        self.employer = None

    def get_hired(self, employer):
        if self.employed:
            raise AlreadyEmployed(f"current employer: {self.employer}")
        self.employer = employer

    def get_skill(self):  # for inheritance
        pass


def main():
    firm = Firm()
    workers = [Worker(i) for i in range(10)]
    pass


if __name__ == "__main__":
    main()
