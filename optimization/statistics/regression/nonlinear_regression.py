"""
Nonlinear regression in the form
 of a numerical optimization function.

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
            + [
                min(data) + ((max(data) - min(data)) / (i + 1))
                for i in reversed(range(1, n_segments))
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


if __name__ == "__main__":
    pwlr_data = Dataset(
        x_data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        y_data=[2, 4, 6, 8, 10, 12, 14, 16, 18],
    )
    segmented_data = pwlr_data.create_segments(n_segments=3)
    print(segmented_data)
    pwlr = PiecewiseLinearRegression(segmented_data)
    pwlr.fit()
    print(pwlr._predict_single(3))
    print(pwlr.predict([1, 2, 3]))
