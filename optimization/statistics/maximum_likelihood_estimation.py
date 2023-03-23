"""
Maximum-Likelihood in the form of a numerical
optimization function.
"""
from typing import List
from abc import ABC, abstractmethod
import math

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
            method="Nelder-Mead"
        )

        self.regression_model = fitted_model
        self.regression_coeff = fitted_model.x

    @abstractmethod
    def predict(self, x_observed):
        """ Makes predictions based on observed data.
        """
        pass


class BernoulliMLE(MaximumLikelihoodEstimatorBase):
    # https://www.analyticsvidhya.com/blog/2022/02/decoding-logistic-regression-using-mle/
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.model = None

    def mle_estimate(self, coeffs: tuple):
        p = coeffs[0]
        n = len(self.y_data)

        # log-likelihood function of Bernoulli data
        return - ((np.log(p) * sum(self.y_data)) + (np.log(1 - p) * (n - sum(self.y_data))))
    
    def mle_regression(self, coeffs: tuple):
        alpha = coeffs[0]
        beta = coeffs[1]

        # log-likelihood function of Bernoulli distribution, with parameter substituted for inverse link transformation.
        return - (sum([self.y_data[i] * (alpha + beta * self.x_data[i]) for i in range(len(self.x_data))]) - sum([np.log(1 + np.exp(alpha + beta * i)) for i in self.x_data]))

    def predict(self, x_observed):
        alpha = self.regression_coeff[0]
        beta = self.regression_coeff[1]
        # Expit (transform) of observed data.
        return [1/(1 + np.exp(-(alpha + (i*beta)))) for i in x_observed]
    

class BinomialMLE(MaximumLikelihoodEstimatorBase):
    # https://www.analyticsvidhya.com/blog/2022/02/decoding-logistic-regression-using-mle/
    def __init__(self, x_data, y_data, n_trials):
        self.x_data = x_data
        self.y_data = y_data
        self.n_trials = n_trials
        self.model = None

    def mle_estimate(self, coeffs: tuple):
        p = coeffs[0]
        n = len(self.y_data)

        # log-likelihood function of Binomial data
        return - (sum([np.log(math.comb(self.n_trials, i)) for i in self.y_data]) + np.log(p)*sum(self.y_data) + np.log(1 - p)*(n*self.n_trials - sum(self.y_data)))
    
    def mle_regression(self, coeffs: tuple):
        alpha = coeffs[0]
        beta = coeffs[1]

        # log-likelihood function of Bernoulli distribution, with parameter substituted for inverse link transformation.
        return - (sum([np.log(math.comb(self.n_trials, i)) for i in self.y_data]) + (sum([self.y_data[i] * (alpha + beta * self.x_data[i]) for i in range(len(self.x_data))]) - sum([np.log(1 + np.exp(alpha + beta * i)) for i in self.x_data])))

    def predict(self, x_observed):
        alpha = self.regression_coeff[0]
        beta = self.regression_coeff[1]
        # Expit (transform) of observed data.
        return [1/(1 + np.exp(-(alpha + (i*beta)))) for i in x_observed]


def bernoulli_fit():
    mle = BernoulliMLE(x_data=[70, 80, 90, 70, 75, 80, 10, 11, 5, 15],
                       y_data=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    mle.fit_estimate()
    print(mle.estimate_model)

    mle.fit_regression()
    print(mle.regression_model)

    print(mle.predict([10, 5, 8, 90]))


def binomial_fit():
    mle = BinomialMLE(x_data=[70, 80, 90, 70, 75, 80, 10, 11, 5, 15],
                      y_data=[7, 8, 9, 5, 6, 4, 5, 6, 8, 9],
                      n_trials=10)
    mle.fit_estimate()
    print(mle.estimate_model)

    mle.fit_regression()
    print(mle.regression_model)

    print(mle.predict([10, 5, 8, 90]))


if __name__ == '__main__':
    mle = BinomialMLE(x_data=[70, 80, 90, 70, 75, 80, 10, 11, 5, 15],
                      y_data=[7, 8, 9, 5, 6, 4, 5, 6, 8, 9],
                      n_trials=10)
    mle.fit_estimate()
    print(mle.estimate_model)

    mle.fit_regression()
    print(mle.regression_model)

    print(mle.predict([10, 5, 8, 90]))