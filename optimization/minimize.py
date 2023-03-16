"""
Function minimization is a commonly used method of optimization.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar


def function1(x):
    """ 
    Univariate optimization.
    f(x) = 3x^2 + 7
    Minimium at x = 0.
    """
    return 3*x**2 + 7


def min_scalar():
    """
    Minimizing a scalar value, using both scalar optimization
    and multivariate optimization.
    """
    x0 = 1

    # Function designed for univariate optimization
    res_min = minimize_scalar(function1, options={'disp': True})

    # Multivariate optimization of 1 univariate function
    res_uni_min = minimize(function1, x0, 
                                  options={'disp': True}, method='Nelder-Mead')
    
    print(res_min)
    print(res_uni_min)


def function2(z):
    """
    Bivariate function optimization.
    f(x,y) = x^2 + 2y^2 - (x + y)
    Minimum at (x,y) = (0.5, 0.25).
    """
    x, y = z
    return x**2 + 2*y**2 - (x + y)


def min_multivariate():
    """
    Multivariate optimization of multivariate function.
    """
    x0 = (1, 1)
    res_multi_min = minimize(function2, x0, 
                                  options={'disp': True}, method='Nelder-Mead')

    print(res_multi_min)


def function3(x, y, z):
    """
    Multivariate function optimization, with two args as constants.
    f(x,y,z) = z^2 + 2y^2 - (z + y) + sin(x)
    Minimum at x = 1.6.
    """
    return z**2 + 2*y**2 - (z + y) + np.sin(x)


def min_multivariate_args():
    """
    Multivariate optimization of multivariate function,
    with two constant arguments.
    """
    x0 = 0
    res_multi_args_min = minimize(function3, x0, 
                             args=(0.5, 0.25),
                             options={'disp': True}, method='Nelder-Mead')

    print(res_multi_args_min)


if __name__ == '__main__':
    min_scalar()
    min_multivariate()
    min_multivariate_args()
    