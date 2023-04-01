"""
Gaussian Mixture Modelling implemented from scratch, using SciPy.
Note that there seem to be problems with convergence - converges on a single
cluster with R = 1 component, with the mean being the mean of 3 components.

Something not quite right with the log-likelihood.
"""

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint


class GaussianMixtureModel:

    def __init__(self, y_data):
        self.y_data = y_data

    def model(self, coeffs: tuple):
        mu = coeffs[0:3]
        sigma = coeffs[3:6]
        mixing_prop = coeffs[6:9]
        n = len(self.y_data)

        # TODO: Function is still not quite correct - issues with distribution of summation over logs.
        # Not sure if I've distributed n properly either.
        fun = - sum([mixing_prop[r] * (-(n/2)* np.log(2*np.pi) - (n/2)*np.log(sigma[r]**2) - (1/(2*sigma[r]**2))*sum((self.y_data - mu[r])**2)) for r in range(len(mixing_prop))])
        print(fun)
        return fun
    
    def fit(self):
        """
        Constraint that mixing proportions must sum to 1.
        Constraint that for all mixing proportions, 0 < proportion < 1.
        Note this is currently only for a gmm of 3 components.
        """
        linear_constraint = LinearConstraint([[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 1]],
                                              [1, 0, 0, 0], [1, 1, 1, 1])
        x0 = (1, 1, 1, 1, 1, 1, 0.3, 0.3, 0.3)

        fitted_model = minimize(
            self.model,
            x0,
            options={"disp": True},
            method="trust-constr",  # Optimization method that allows constraints
            constraints=linear_constraint,
        )

        print(fitted_model)
        self.estimate_model = fitted_model
        self.estimate_coeff = fitted_model.x


if __name__ == '__main__':
    data = [
        1, 1, 1, 1, 1,
        50, 50, 50, 50, 50,
        100, 100, 100, 100, 100
        ]
    
    gmm = GaussianMixtureModel(data)
    gmm.fit()
