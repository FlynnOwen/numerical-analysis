"""
Funcitons related to the field of statistics written as
numerical functions to be minimzed by scipy.
"""
from typing import List
from abc import ABC, abstractmethod

from scipy.optimize import minimize
import numpy as np


class TimeSeriesModel(ABC):
    def __init__(self, training_data: List):
        self.training_data = training_data
        self.model = None

    @staticmethod
    def _mae_ar(y: List, yhat: List):
        """
        Calculate the mean absolute error,
        given predicted and actual values.

        Args:
            y (list): actual values
            yhat (list): predicted values

        Returns:
            mae: mean absolute error
        """
        absolute_epsilon = [np.abs(y[i] - yhat[i - 1]) for i in range(1, len(yhat))]
        return sum(absolute_epsilon) / len(absolute_epsilon)

    @abstractmethod
    def time_series_model(self, coeffs: tuple):
        pass

    def fit_model(self, starting_value: int = 1):
        """
        Multivariate optimization of multivariate function.
        """
        fitted_model = minimize(
            self.time_series_model,
            starting_value,
            options={"disp": True},
            method="Nelder-Mead",
        )

        self.model = fitted_model
        self.coeff = fitted_model.x

    @abstractmethod
    def predict(self, horizon):
        """
        Generates predicted future values.
        """
        pass


class AutoRegressionOrder1(TimeSeriesModel):
    def __init__(self, training_data: List):
        self.training_data = training_data
        self.model = None

    def time_series_model(self, rho):
        """Trains an order 1 auto-regressive model.

        Args:
            rho (int): coefficient for auto-regressive parameter.
        """
        weighted_data = [rho * i for i in self.training_data]
        return self._mae_ar(self.training_data, weighted_data)

    def predict(self, horizon):
        """
        Generates predicted future values.
        """
        max_value = self.training_data[-1]
        predicted_values = []
        for _ in range(horizon):
            max_value = max_value * self.coeff
            predicted_values.append(max_value)

        return predicted_values


if __name__ == "__main__":
    ar = AutoRegressionOrder1([1, 2, 3, 4, 5, 6, 7, 8, 9])
    ar.fit_model()

    print(ar.model)
    print(ar.predict(5))
