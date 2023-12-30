"""
Linear regression in the form of a numerical
optimization function.
"""
from typing import List

from scipy.optimize import minimize
import numpy as np


class LinearRegression:
    def __init__(self, x_data: list[float], y_data: list[float]):
        self.x_data = x_data
        self.y_data = y_data
        self.model = None

    @staticmethod
    def _mae(y: list[float], yhat: list[float]):
        absolute_epsilon = [np.abs(y[i] - yhat[i]) for i in range(len(yhat))]
        return sum(absolute_epsilon) / len(absolute_epsilon)

    def _linear_regression(self, coeffs: tuple[int, int]):
        alpha = coeffs[0]
        beta = coeffs[1]
        weighted_data = [alpha + (beta * i) for i in self.x_data]
        return self._mae(self.y_data, weighted_data)

    def fit(self, starting_value: tuple[int] = (1, 1)):
        fitted_model = minimize(
            self._linear_regression,
            starting_value,
            options={"disp": True},
            method="Nelder-Mead",
        )

        self.model = fitted_model
        self.coeff = fitted_model.x

    def predict(self, x_observed: List):
        alpha = self.coeff[0]
        beta = self.coeff[1]
        return [alpha + (i * beta) for i in x_observed]


if __name__ == "__main__":
    ar = LinearRegression(
        x_data=[1, 2, 3, 4, 5, 6, 7, 8, 9], y_data=[2, 4, 6, 8, 10, 12, 14, 16, 18]
    )
    ar.fit()

    print(ar.model)
    print(ar.predict([11, 12, 13]))
