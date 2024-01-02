"""
Nonlinear regression in the form
 of a numerical optimization function.

 This includes:
 - Piecewise constant
 - Piecewise linear
 - Piecewise continuous linear
 - Piecewise linear basis
"""
from typing import List

from scipy.optimize import minimize
import numpy as np

from optimization.statistics.regression._linear_regression import LinearRegression


class NonLinearRegression:
    def __init__(self):
        pass
