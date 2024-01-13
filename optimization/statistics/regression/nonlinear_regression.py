"""
Nonlinear regression in the form
 of a numerical optimization function.

https://bookdown.org/tpinto_home/Beyond-Linearity/piecewise-regression-and-splines.html

 This includes:
 - Piecewise constant
 - Piecewise linear
 - Piecewise continuous linear
 - Piecewise linear basis
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from scipy.optimize import minimize, curve_fit

from optimization.statistics.regression.linear_regression import LinearRegression


@dataclass
class Dataset:
    """
    Encapsulation of data needed for linear regression.
    """

    x_data: list[float]
    y_data: list[float]
    model: None = None
    coeff: None = None

    @staticmethod
    def _uniform_split(data: list[float], n_segments: int):
        """
        Generate uniform value-based bounds that define how
        to split a list of floats into.
        """
        return (
            [-np.inf]
            + list(np.linspace(start=min(data), stop=max(data), num=n_segments + 1))[
                1:-1
            ]
            + [np.inf]
        )

    def _conditional_sort(self, list1: list[float], list2: list[Any]):
        """
        Sort list2 conditional on the sorting of list1.
        """
        return zip(*sorted(zip(list1, list2)))

    def create_segments(self, n_segments: int) -> list[SegmentedDataset]:
        # TODO: This is ugly - refactor
        segment_collection = []

        segment_bounds = self._uniform_split(self.x_data, n_segments)
        x_data_sorted, y_data_sorted = self._conditional_sort(self.x_data, self.y_data)
        segment_x_data = []
        segment_y_data = []
        j = 1

        for i in range(len(x_data_sorted)):
            if x_data_sorted[i] <= segment_bounds[j]:
                segment_x_data.append(x_data_sorted[i])
                segment_y_data.append(y_data_sorted[i])
            else:
                segment = SegmentedDataset(
                    x_data=segment_x_data,
                    y_data=segment_y_data,
                    lower_bound=segment_bounds[j - 1],
                    upper_bound=segment_bounds[j],
                )
                segment_collection.append(segment)
                segment_x_data = [x_data_sorted[i]]
                segment_y_data = [y_data_sorted[i]]

                j += 1
        segment = SegmentedDataset(
            x_data=segment_x_data,
            y_data=segment_y_data,
            lower_bound=segment_bounds[j - 1],
            upper_bound=segment_bounds[j],
        )
        segment_collection.append(segment)

        return segment_collection


@dataclass
class SegmentedDataset(Dataset):
    """
    Encapsulation of data needed for piecewise linear regression.
    """

    lower_bound: float | None = None
    upper_bound: float | None = None


class PiecewiseLinearRegression:
    def __init__(self, data: list[SegmentedDataset]):
        self.data: list[SegmentedDataset] = data

    def fit(self) -> None:
        """
        Fit a Linear Regression model for each segment.
        """
        for dataset in self.data:
            lr = LinearRegression(x_data=dataset.x_data, y_data=dataset.y_data)
            lr.fit()
            dataset.model = lr.model
            dataset.coeff = lr.coeff

    def _predict_single(self, x_single: float):
        """
        Iterate through segmented x data and find which
        segment this value belongs within.
        """
        i = 0
        while True:
            if (
                x_single >= self.data[i].lower_bound
                and x_single < self.data[i].upper_bound
            ):
                alpha = self.data[i].coeff[0]
                beta = self.data[i].coeff[1]

                return alpha + (x_single * beta)
            else:
                i += 1

    def predict(self, x_list: list[float]):
        return [self._predict_single(x_single=x_single) for x_single in x_list]


#TODO: Below is still in dev
class ContPiecewiseLinearRegression:
    def __init__(self, data: Dataset):
        self.data = data

    @property
    def x_data(self):
        print(self.data)
        return self.data.x_data

    @property
    def y_data(self):
        return self.data.y_data

    def objective(self, coeffs: tuple[float], breakpoints: tuple[float]):
        # Objective function to minimize
        slopes = coeffs[: len(breakpoints) - 1]
        y_intercepts = coeffs[len(breakpoints) + 1 :]
        x = np.array(self.x_data)
        print(x)
        print(slopes)
        print(y_intercepts)
        y_predicted = np.piecewise(
            x, [x < b for b in breakpoints], slopes * x + y_intercepts
        )
        print(y_predicted)
        return
        return np.sum((y_predicted - self.y_data) ** 2)

    def fit(self) -> None:
        """
        Fits a piecewise linear function that is constraint to be
        continuous.
        """
        breakpoints = self.data._uniform_split(self.x_data, 3)
        print(breakpoints)
        # Initial guess for the parameters
        initial_params = np.zeros((len(breakpoints) - 1) * 2)
        print(initial_params)

        # Minimize the objective function
        return minimize(
            self.objective, initial_params, args=breakpoints, method="L-BFGS-B"
        )

    def predict(self, x_list: list[float]):
        pass


def piecewise_regression_objective(params, x, y, breakpoints):
    """
    Objective function for piecewise regression using np.piecewise and scipy.optimize.minimize.

    Parameters:
    - params: Parameters to be optimized (slopes and intercepts for each segment)
    - x: Input data points
    - y: Actual y-values
    - breakpoints: List of x-values where linear segments change

    Returns:
    - Sum of squared differences between predicted and actual y-values
    """

    num_breakpoints = len(breakpoints)
    slopes = params[:num_breakpoints]
    intercepts = params[num_breakpoints:]
    y_predicted = np.piecewise(x, [x < breakpoints[b] for b in range(len(breakpoints))],
                                  [lambda x: slopes[b] * x + intercepts[b] for b in range(len(breakpoints))])
    return np.sum((y_predicted - y) ** 2)


def piecewise_regression_custom(x, *params):
    num_segments = 2

    # Initial guess for breakpoints (evenly spaced)
    breakpoints = np.linspace(np.min(x), np.max(x), num_segments + 1)[
        :-1
    ]
    num_breakpoints = len(breakpoints)
    slopes = params[:num_breakpoints]
    intercepts = params[num_breakpoints:]
    return np.piecewise(x, [x < breakpoints[b] for b in range(len(breakpoints))],
                             [lambda x: slopes[b] * x + intercepts[b] for b in range(len(breakpoints))])


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(
        x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0]
    )

if __name__ == "__main__":
    pwlr_data = Dataset(
        x_data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        y_data=[2, 4, 6, 8, 10, 12, 14, 16, 18],
    )
    segmented_data = pwlr_data.create_segments(n_segments=3)
    pwlr = PiecewiseLinearRegression(segmented_data)
    pwlr.fit()
    print(pwlr._predict_single(3))
    print(pwlr.predict([1, 2, 3]))

    # Example usage:
    x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=float)
    y_data = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
        ]
    )

    num_segments = 2
    # Initial guess for the parameters
    initial_params = np.zeros(2 * num_segments)

    # Initial guess for breakpoints (evenly spaced)
    breakpoints_guess = np.linspace(np.min(x_data), np.max(x_data), num_segments + 1)[
        :-1
    ]
    result = curve_fit(piecewise_linear, x_data, y_data)

    print("Curve fit Result: \n")
    print(result)

    print("Custom Result: \n")
    print(minimize(fun=piecewise_regression_objective,
                   x0=initial_params,
                   args=(x_data, y_data, breakpoints_guess)))
    print(curve_fit(piecewise_regression_custom, x_data, y_data, p0=(0,0,0,0)))

