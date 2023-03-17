"""
In this instance, I refer to fixed order integration as integration over 1, 2,
or 3 variables.
"""
from dataclasses import dataclass

from scipy import integrate
import numpy as np


@dataclass
class Bounds:
    lower: float
    upper: float


def function1(x):
    """
    A basic univariate function to integrate defined as:
    f(x) = 3x^2 + sin(x)
    Integral should be:
    f'(x) = x^3 - cos(x)
    """
    return 3*(x**2) + np.sin(x)


def integral_uni_1():
    """
    Analytical evaluation:
    f'(x) = (1 - 0.54) - (0 - 1) = 0.46 + 1 = 1.46
    """
    bounds = Bounds(0, 1)
    res = integrate.quad(function1, bounds.lower, bounds.upper)

    print(res)


def function2(x, y):
    """
    A bivariate function to integrate defined as:
    f(x, y) = 3x^2 + 2y + 4xy + 2
    Integral should be:
    f'x(x, y) = x^3 + 2xy + 2yx^2 + 2x
    f'xy(x ,y) = yx^3 + xy^2 + x^2y^2 + 2xy
    """
    return 3*(x**2) + 2*y + 4*x*y + 2


def integral_bi_1():
    """
    Analytical evaluation:
    f'(x ,y) = (1 + 1 + 1 + 2) - (0) = 5
    """
    x_bounds = Bounds(0, 1)
    y_bounds = Bounds(0, 1)
    res = integrate.dblquad(function2,
                            x_bounds.lower,
                            x_bounds.upper,
                            y_bounds.lower,
                            y_bounds.upper)

    print(res)


if __name__ == '__main__':
    integral_uni_1()
    integral_bi_1()
