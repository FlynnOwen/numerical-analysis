# Optimization Algorithms

## Nelder-Mead
Nelder-Mead is considered a *direct* search method (based on function comparison) and is often applied to nonlinear optimization problems for which derivatives may not be known.

It's primary method is creating a simplex using a number of parameter values, evaluating each of them, and then adjusting the topology of the simplex depending on the evaluations.

It consists of a number of operations:

- Shrink
- Contract
- Reflect
- Expand

<p align="center">
<img src="../img/nelder_iteration.png"/>
</p>