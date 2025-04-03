from typing import Callable
from dataclasses import dataclass
from math import sqrt


@dataclass
class Result:
    position: list[float]
    eval_func: float
    eval_gradient: float
    iterations: int


class GradientDescent:

    def __init__(
        self,
        func: Callable,
        gradient_func: Callable,
        ) -> None:
        self.func = func
        self.gradient_func = gradient_func

    def _norm(self, vector: list[float]) -> float:
        return sqrt(sum([x**2 for x in vector]))

    def minimize(
        self,
        start: list[float],
        learning_rate: float,
        max_iter: int,
        gradient_threshold: float
        ) -> Result:
        iteration = 0
        position = start
        while iteration < max_iter:
            iteration += 1
            gradient = self.gradient_func(*position)
            for idx in range(len(position)):
                position[idx] = position[idx] - (learning_rate * gradient[idx])
            if self._norm(gradient) <= abs(gradient_threshold):
                break
        return Result(
            position,
            self.func(*position),
            gradient,
            iteration
            )

    def maximize(
        self,
        start,
        learning_rate: float,
        max_iter: int,
        gradient_threshold: float
        ) -> Result:
        iteration = 0
        position = start
        while iteration < max_iter:
            print(position)
            gradient = self.gradient_func(*position)
            for idx in range(len(position)):
                position[idx] = position[idx] + (learning_rate * gradient[idx])
            if self._norm(gradient) <= abs(gradient_threshold):
                break
        return Result(
            position,
            self.func(*position),
            gradient,
            iterations
            )


def function(x, y):
    return x**2 + y**2


def gradient(x, y):
    return [2*x, 2*y]


optimizer = GradientDescent(
    func=function,
    gradient_func=gradient
)


result = optimizer.minimize(
    start=[3, 4],
    learning_rate=0.05,
    max_iter=100,
    gradient_threshold=0.02
    )

print(result)
