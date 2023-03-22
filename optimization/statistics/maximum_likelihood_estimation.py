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
        self.estimate_model = None
        self.regression_model = None

    @abstractmethod
    def mle_estimate(self, coeffs: tuple):
       """ Estimates maximum likelihood parameters
       """
       pass

    @abstractmethod
    def mle_regression(self, coeffs: tuple):
       """ Estimates alpha and beta coefficients used in regression
       """
       pass

    def fit_estimate(self, starting_value: tuple[int] = (0.5, )):
        """ Fits MLE given data.
        """
        fitted_model = minimize(
            self.mle_estimate,
            starting_value,
            options={"disp": True},
            method="Nelder-Mead"
        )

        self.estimate_model = fitted_model
        self.estimate_coeff = fitted_model.x

    def fit_regression(self, starting_value: tuple[int] = (0.5, 0.5)):
        """ Fits regression model using x and y data.
        """
        fitted_model = minimize(
            self.mle_regression,
            starting_value,
            options={"disp": True},
            method="Nelder-Mead"
        )

        self.regression_model = fitted_model
        self.regression_coeff = fitted_model.x

    @abstractmethod
    def predict(self, x_observed):
        """ Makes predictions based on observed data.
        """
        pass
