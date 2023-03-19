"""
Integrating with n > 3 variates.
"""
from scipy import integrate

from fixed_order_integrals import Bounds


def function1(x, y, z, b):
    """
    A multivariate function to integrate defined as:
    f(x, y, z, b) = 2x + 2y + 2z + 2b + 16xyzb
    Integral should be:
    f'x(x, y, z, b) = x^2 + 2xy + 2xz + 2xb + 8x^2yzb
    f'xy(x, y, z, b) = x^2y + x^2y + 2xyz + 2xyb + 4zx^2y^2b
    f'xyz(x, y, z, b) = x^2yz + x^2yz + xyz^2 + 2xyzb + 2x^2y^2z^2b
    f'xyzb(x, y, z, b) = x^2yzb + x^2yzb + xyz^2b + xyzb^2 + x^2y^2z^2b^2
    """
    return 2 * x + 2 * y + 2 * z + 2 * b + 16 * x * y * z * b


def integral_n_1():
    """
    Analytical evaluation:
    f'(x ,y, z) = (4 + 4 + 8 + 8 + 16) - (0) = 40
    """
    x_bounds = Bounds(0, 1)
    y_bounds = Bounds(0, 1)
    z_bounds = Bounds(0, 2)
    b_bounds = Bounds(0, 2)
    res = integrate.nquad(
        function1,
        [
            [x_bounds.lower, x_bounds.upper],
            [y_bounds.lower, y_bounds.upper],
            [z_bounds.lower, z_bounds.upper],
            [b_bounds.lower, b_bounds.upper],
        ],
    )

    print(res)


if __name__ == "__main__":
    integral_n_1()
