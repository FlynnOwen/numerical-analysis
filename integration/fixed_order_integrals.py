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
    A basic function to integrate defined as:
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


if __name__ == '__main__':
    integral_uni_1()
