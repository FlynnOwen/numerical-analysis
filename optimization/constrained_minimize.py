"""
Function minimization is a commonly used method of optimization.
Often we want to apply additional constraints to the optimization problem.

Example constraints might be:
- Bounds (e.g must be positive / negative)
- Variate constraints (e.g variates must sum to 1)
- Linearity
"""
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint


def function2(z):
    """
    Bivariate function optimization.
    f(x,y) = x^2 + 2y^2 - (x + y)
    Minimum at (x,y) = (0.5, 0.25).
    Lower bounds defined at (0.75, 0.5) should mean
    new minimum at (x,y) = (0.75, 0.5)
    """
    x, y = z
    return x**2 + 2 * y**2 - (x + y)


def min_multivariate_bounds():
    """
    Multivariate optimization of multivariate function,
    with variable bounds.
    """
    x0 = (1, 1)
    bounds = Bounds([0.75, 0.5], [2, 3])
    res_multi_min = minimize(
        function2, x0, options={"disp": True}, method="Nelder-Mead", bounds=bounds
    )

    print(res_multi_min)


def min_multivariate_lin_conts():
    """
    Multivariate optimization of multivariate function,
    with a linear constraint on variates.
    Constraints are:
    x + 2y <= 1
    2x + y = 1
    """
    linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])
    x0 = (1, 1)

    res_multi_min = minimize(
        function2,
        x0,
        options={"disp": True},
        method="trust-constr",  # Optimization method that allows constraints
        constraints=linear_constraint,
    )

    print(res_multi_min)


def min_multivariate_nonlin_conts():
    """
    Multivariate optimization of multivariate function,
    with a non-linear constraint on variates.
    Constraints are:
    x^2 + y <= 1
    x^2 - y <= 1

    These must be defined as:
    - A function
    - A Jacobian (first order derivative)
    - A combination  of Hessians (second order derivatives)
    """

    def cons_f(x):
        return [x[0] ** 2 + x[1], x[0] ** 2 - x[1]]

    def cons_J(x):
        return [[2 * x[0], 1], [2 * x[0], -1]]

    def cons_H(x, v):
        return v[0] * np.array([[2, 0], [0, 0]]) + v[1] * np.array([[2, 0], [0, 0]])

    nonlinear_constraint = NonlinearConstraint(
        cons_f, [-np.inf, -np.inf], [1, 1], jac=cons_J, hess=cons_H
    )
    # Option to instead approximate the Hessian using finite differences: hess='2-point'
    # Jacobian can also be approximated using finite differences: jac='2-point', hess=BFGS()

    x0 = (1, 1)

    res_multi_min = minimize(
        function2,
        x0,
        options={"disp": True},
        method="trust-constr",  # Optimization method that allows constraints
        constraints=nonlinear_constraint,
    )

    print(res_multi_min)


if __name__ == "__main__":
    min_multivariate_bounds()
    min_multivariate_lin_conts()
    min_multivariate_nonlin_conts()
