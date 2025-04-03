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

## Gradient Descent

Gradient Descent is considered a 'gradient-based' optimization method. Using a gradient, it will traverse in the direction that the gradient is most negative for minimisation (or most positive for maximisation) problems.

The function must be *defined* and *differentiable*. 

The general formula is:

$x_{t+1} = x_t + \alpha \nabla f(x_t),$

Where:
- $\nabla f(x_t)$ is the gradient of $f$ evaluated with variables $x_t$
- $\alpha$ is the *learning rate*