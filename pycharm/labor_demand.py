import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

A = 2.0  # constant factor
alpha = 0.75  # labor share
beta = 0.25  # capital share
wage = 1.2
rate = 0.8
price = 100

def find_optimal_output(variables: list[wage: float, rate: float, price: float], constants: list[A: float, alpha: float, beta: float]):
    # return constant_rts: Bool, optimal_output: float
    wage, float, price = variables
    A, alpha, beta = constants
    rts = alpha+beta # stands for return to scale. if rts<1 -> decreasing returns to scale. if rts>1 -> increasing returns to scale
    if rts == 1: # constant returns to scale -> zero economic profit :(
        return True # TODO: find optimal_output for constant rts
    c_term = 1/rts
    a_term = A**(-1/rts)
    w_term = wage**(alpha/rts)  # double check
    r_term = rate**(beta/rts)  # double check
    alpha_beta_term = ((alpha/beta)**(beta/rts)) + ((beta/alpha)**(alpha/rts))  # double check
    main_term = price*((c_term*a_term*w_term*r_term*alpha_beta_term)**-1)  # double check
    return False, main_term**(rts/1-rts)

pass
