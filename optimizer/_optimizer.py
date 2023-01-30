"""Optimizer module"""

from typing import Callable, Optional
from numpy.random import randn
from scipy.optimize import minimize

class Optimizer:
    """Gradient descent optimization of a cost function"""

    def __init__(self, stepsize = 0.01):
        self.stepsize = stepsize


    def step(self, grad, *params):
        """Updates the cost function with a gradient descent step."""
        params -= self.stepsize * grad(params)
        return params


    def GD_minimize(self, grad: Callable, params, epochs: int = 100):
        """Minimize a given cost function using gradient descent."""

        for _ in range(epochs):
            params = self.step(grad, params)
        return params


    def LBFGSB_minimize(self, cost: Callable, grad: Callable, params):
        """Minimize a given cost function using L-BFGS-B method."""
        opt_result = minimize(cost, params, args = (), method = 'L-BFGS-B', jac = grad)
        return opt_result.x # optimized params
        

    def minimize(self,
                cost: Callable,
                grad: Callable,
                method: str,
                num_params: int,
                init_params = None,
                **kwargs):
        """Minimize the cost function using different methods."""

        if init_params is None:           
            params = randn(num_params)

        if method == 'GD':
            return self.GD_minimize(grad, params, **kwargs)

        if method == 'L-BFGS-B':
            return self.LBFGSB_minimize(cost, grad, params, **kwargs)

        raise ValueError("Argument 'method' must be either 'GD' or 'L-BFGS-B'.")
