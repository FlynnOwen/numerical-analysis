from scipy.optimize import minimize

from nelder_mead import NelderMead


def sum_to_twelve(values: list[float]) -> float:
    return abs(12 - sum(values))

def rosenbrock(x: list[float]) -> float:
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def test_func(x: list[float]) -> float:
    return x[0]**2 + x[1]**2

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

optim = NelderMead(
    func=rosenbrock,
    #starting_point=[
    #    [-1.0, -.5],
    #    [0, .5],
    #    [1.0, 0]
    #    ],
    starting_point=[
        [ 0.61157155, -0.27257958],
        [0.8918778, 0.6791299],
        [0.4700612, 0.7962842]
        ]
    )

out = optim.minimize()
print(out)

#def generate_simplex(x0, epsilon=0.05):
#    n = len(x0)
#    simplex = [x0.copy()]
#    for i in range(n):
#        vertex = x0.copy()
#        vertex[i] += epsilon * (abs(x0[i]) + 1)
#        simplex.append(vertex)
#    return simplex
#print(generate_simplex([-1.2, 1.0]))
#
#out = minimize(
#    rosenbrock,
#    x0=(-1.2, 1.0),
#    method="Nelder-Mead")
#print(out)