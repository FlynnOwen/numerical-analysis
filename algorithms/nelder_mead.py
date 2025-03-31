from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from functools import total_ordering


@dataclass
@total_ordering
class Vertex:
    point: list[float]
    score: float | None = None

    def __eq__(self, vertex: Vertex) -> bool:
        return self.score == vertex.score

    def __lt__(self, vertex: Vertex) -> bool:
        if self.score is None or vertex.score is None:
            raise ValueError("score must be calculated prior to comparison.")
        return self.score < vertex.score

    def __add__(self, vertex: Vertex):
        return [self.point[i] + vertex.point[i] for i in range(len(self.point))]

    def __getitem__(self, key):
        return self.point[key]

    def __len__(self):
        return len(self.point)


@dataclass
class Simplex:
    vertices: list[Vertex]

    def sort(self):
        self.vertices.sort()

    def all_but_worst(self, points: bool = True) -> list[Vertex] | list[list[float]]:
        if points:
            return [vertex.point for vertex in self.vertices[:-1]]
        else:
            return self.vertices[:-1]

    def all_but_best(self, points: bool = True) -> list[Vertex]:
        if points:
            return [vertex.point for vertex in self.vertices[1:]]
        else:
            return self.vertices[1:]

    @property
    def worst(self) -> Vertex:
        return self.vertices[-1]

    @worst.setter
    def worst(self, vertex: Vertex):
        self.vertices[-1] = vertex

    @property
    def second_worst(self) -> Vertex:
        return self.vertices[-2]

    @property
    def best(self) -> Vertex:
        return self.vertices[0]

    @best.setter
    def best(self, vertex: Vertex):
        self.vertices[0] = vertex

    def __iter__(self):
        return iter(self.vertices)

    def __len__(self):
        return len(self.vertices)

    def centroid(self) -> Vertex:
        return Vertex([sum(axis)/(len(self) - 1) for axis in zip(*self.all_but_worst())])

class NelderMead:
    """
    Perform Nelder-Mead optimization.
    """
    def __init__(
        self,
        func: Callable,
        starting_point: list[list[float]],
        *,
        alpha: float = 1,
        gamma: float = 2,
        beta: float = 0.5,
        delta: float = 0.5,
        epsilon: float = 0.1
        ) -> None:
        self.func = func
        self.simplex: Simplex = self._init_simplex_points(starting_point)
        self._alpha = alpha
        self._gamma = gamma
        self._beta = beta
        self._delta = delta

    def _init_simplex(
        self,
        starting_point: list[float],
        epsilon: float = 0.05
        ) -> Simplex:
        """
        Create a simplex of size n + 1, where n = |starting_point|.

        The formula for vertex i is
        x_i = x_0 + (e_i * epsilon_i),
        Where e_i is the unit vector of the i_th coordinate axis.
        """
        simplex = []
        simplex.append(Vertex(starting_point))
        for i in range(len(starting_point)):
            point = starting_point.copy()
            point[i] += epsilon
            simplex.append(Vertex(point))
        simplex_ = Simplex(simplex)
        self._evaluate(simplex_)

        return simplex_

    def _init_simplex_points(
        self,
        starting_point: list[list[float]],
        ) -> Simplex:
        """
        Create a simplex of size n + 1, where n = |starting_point|.

        The formula for vertex i is
        x_i = x_0 + (e_i * epsilon_i),
        Where e_i is the unit vector of the i_th coordinate axis.
        """
        simplex = [Vertex(point) for point in starting_point]
        simplex_ = Simplex(simplex)
        self._evaluate(simplex_)

        return simplex_

    def _evaluate(
        self,
        simplex: Vertex | Simplex
        ) -> None:
        """
        Evaluate the provided function, using a particular vertex.
        """
        if isinstance(simplex, Vertex):
            simplex = [simplex]
        for vertex in simplex:
            vertex.score = self.func(vertex)

    def _reflect(
        self,
        centroid: Vertex,
        worst: Vertex
        ) -> Vertex:
        """
        Calculate the reflection.
        x_r = x_c + a(x_c  - x_{n+1})

        Where:
        - x_r is the reflection
        - x_c is the centroid

        Accepted if func(x_r) < func(x_1)
        """
        point = []
        for idx in range(len(centroid)):
            axis = centroid[idx] + self._alpha * (centroid[idx] - worst[idx])
            point.append(axis)
        reflection = Vertex(point)
        self._evaluate(reflection)

        return reflection

    def _expand(
        self,
        centroid: Vertex,
        reflection: Vertex
        ) -> Vertex:
        """
        Calculate the expansion.
        Only performed if func(`reflection`) < func(x1)

        x_e = x_c + gamma(xr - xc)

        Accepted if func(x_e) < func(x_r)
        """
        point = []
        for idx in range(len(centroid)):
            axis = centroid[idx] + self._gamma * (reflection[idx] - centroid[idx])
            point.append(axis)
        expansion = Vertex(point)
        self._evaluate(expansion)

        return expansion

    def _outside_contract(
        self,
        centroid: Vertex,
        reflection: Vertex
        ) -> Vertex:
        """
        Performed if f(x_r) >= f(x_{n+1})

        x_o = x_c + beta(x_r - x_c)
        """
        point = []
        for idx in range(len(centroid)):
            axis = centroid[idx] + self._beta * (reflection[idx] - centroid[idx])
            point.append(axis)
        outside_contraction = Vertex(point)
        self._evaluate(outside_contraction)

        return outside_contraction

    def _inside_contract(
        self,
        centroid: Vertex,
        reflection: Vertex,
        worst: Vertex
        ) -> Vertex:
        """
        Performed if f(x_r) < f(x_{n+1})

        x_i = x_c + beta(x_r - x_{n+1})
        """
        point = []
        for idx in range(len(centroid)):
            axis = centroid[idx] + self._beta * (reflection[idx] - worst[idx])
            point.append(axis)
        inside_contraction = Vertex(point)
        self._evaluate(inside_contraction)

        return inside_contraction

    def _shrink(self, best: Vertex) -> None:
        """
        Performed if both contractions fail.
        We replace all points (x_i) except the best point (x_1).

        x_s = x_1 + delta(x_i - x1)
        """
        vertices = []
        for vertex in self.simplex.all_but_best():
            point = []
            for idx in range(len(vertex)):
                point.append(best[idx] + self._delta * (vertex[idx] - best[idx]))
            vertices.append(Vertex(point))

        self.simplex.vertices = [self.simplex.best] + vertices

    def minimize(self) -> Vertex:
        """
        - Evaluate points
        - Order them best to worst
        - Calculate centroid of all points except x_{n+1} (x_c)
        - Compute reflection (x_r)
        - If f(x_1) <= f(x_r) < f(x_n), then replace x_{n+1} with x_r and restart
        - Else If f(x_r) < f(x_1), compute the expansion
            - If f(x_e) < f(x_r), then replace x_{n+1} with x_e and restart
            - Else replace x_{n+1} with x_r and restart
        - Else If f(x_n) <= f(x_r) < f(x_{n+1}), compute outside contraction (x_o)
            - If f(x_o) < f(x_r), then replace x_{n+1} with x_o and restart
            - Else <shrink>
        - Else If f(x_r) >= f(x_{n+1}), compute inside contraction (x_i)
            - If f(x_i) < f(x_{n+1}), then replace x_{n+1} with x_i and restart
            - Else replace x_{n+1} with x_r and restart
            - Else <shrink>
        - <shrink>: Replace all points except x_1 with shrunk value
        """
        # TODO: Break if evaluates successfully within some tolerance.
        from time import sleep
        iter = 0
        while self.simplex.best.score > 0.001 and iter < 10000:
            iter += 1
            self.simplex.sort()
            centroid = self.simplex.centroid()
            self._evaluate(centroid)
            reflection = self._reflect(centroid, self.simplex.worst)
            if self.simplex.best <= reflection < self.simplex.second_worst:
                print("reflect")
                self.simplex.worst = reflection
            elif reflection < self.simplex.best:
                expansion = self._reflect(centroid, reflection)
                if expansion < reflection:
                    print("expand")
                    self.simplex.worst = expansion
                else:
                    print("reflect")
                    self.simplex.worst = reflection
            elif self.simplex.second_worst <= reflection < self.simplex.worst:
                outside_contraction = self._outside_contract(centroid, reflection)
                if outside_contraction < reflection:
                    print("outside contract")
                    self.simplex.worst = outside_contraction
                else:
                    print("shrink")
                    self._shrink(self.simplex.best)
            elif reflection >= self.simplex.worst:
                inside_contraction = self._inside_contract(centroid, reflection, self.simplex.worst)
                if inside_contraction < self.simplex.worst:
                    print("inside contract")
                    self.simplex.worst = inside_contraction
                else:
                    print("shrink")
                    self._shrink(self.simplex.best)
            else:
                raise Exception
            self._evaluate(self.simplex)

        return self.simplex
