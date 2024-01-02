"""
Utility functions related to regression.
"""
import numpy as np


def mae(y: list[float], yhat: list[float]):
    absolute_epsilon = [np.abs(y[i] - yhat[i]) for i in range(len(yhat))]
    return sum(absolute_epsilon) / len(absolute_epsilon)
