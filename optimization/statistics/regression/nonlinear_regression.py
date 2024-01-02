"""
Nonlinear regression in the form
 of a numerical optimization function.

 This includes:
 - Piecewise constant
 - Piecewise linear
 - Piecewise continuous linear
 - Piecewise linear basis
"""
from typing import List
from itertools import batched
from dataclasses import dataclass
from functools import total_ordering

from optimization.statistics.regression._linear_regression import LinearRegression


@dataclass
@total_ordering
class Segment:
    def __init__(self, x_data: list[float], y_data: list[float]):
        self.x_data = x_data
        self.y_data = y_data
        self.model = None

    def __lt__(self):
        pass

    def __eq__(self):
        pass


class PiecewiseLinearRegression:
    # TODO: Need to force that x_data (and y_data conditionally
    # on x_data) is sorted ascending in order to segment properly.
    def __init__(self, x_data: list[float], y_data: list[float], n_segments: int):
        self.x_data = x_data
        self.y_data = y_data
        self.n_segments = n_segments

        self.fitted_segments: list[LinearRegression] = []

    def _conditional_sort(self, list1, list2):
        """
        Sort list2 conditional on the sorting of list1.
        """
        return zip(*sorted(zip(list1, list2)))

    def _segment_data(self, data: list[float], n_segments: int) -> list[tuple[int]]:
        """
        Splits the provided dataset into a defined
        number of segments, with an equal number
        of observations in each segment.
        """
        return list(batched(self.data, n_segments))

    def _segment_xy(self):
        """
        Sorts Y conditional on X, and then splits the data into a number of segments.
        """
        x_data_sorted, y_data_sorted = self._conditional_sort(self.x_data, self.y_data)
        self._x_data_segmented = self._segment_data(x_data_sorted, self.n_segments)
        self._y_data_segmented = self._segment_data(y_data_sorted, self.n_segments)

    def fit(self) -> None:
        """
        Fit a Linear Regression model for each segment.
        """
        for i in self.n_segments:
            lr = LinearRegression(
                x_data=self._x_data_segmented[i], y_data=self._y_data_segmented[i]
            )
            lr.fit()
            self.fitted_segments.append(lr)

    def predict(self, x_observed: List):
        alpha = self.coeff[0]
        beta = self.coeff[1]
        return [alpha + (i * beta) for i in x_observed]
