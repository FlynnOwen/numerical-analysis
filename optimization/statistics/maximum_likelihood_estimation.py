"""
Maximum-Likelihood in the form of a numerical
optimization function.
"""
from typing import List
from abc import ABC, abstractmethod

from scipy.optimize import minimize
import numpy as np

class MaximumLikelihoodEstimatorBase(ABC):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.model = None

    @abstractmethod
    def _regression(self, coeffs: tuple):
       pass

    @abstractmethod
    def _calculate_error(self, y: List, yhat: List):
        pass

    def fit(self, starting_value: tuple[int] = (1, 1)):
        fitted_model = minimize(
            self._regression,
            starting_value,
            options={"disp": True},
            method="Nelder-Mead"
        )

        self.model = fitted_model
        self.coeff = fitted_model.x

    @abstractmethod
    def predict(self, x_observed: List):
        pass
