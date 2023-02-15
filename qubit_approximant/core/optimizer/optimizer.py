"""
Optimizers for our cost function.

Classes
-------
Optimizer:
    Base class for the optimization of a circuit with fixed number of layers.
BlackBoxOptimizer:
    This optimizer uses scipy.minimize as a black box optimizer.
GDOptimizer:
    Gradient descent optimizer.
AdamOptimizer:
    A Method for Stochastic Optimization optimizer.
"""

from abc import ABC, abstractmethod
from typing import Callable

from numpy import ndarray, zeros_like, sqrt
from scipy.optimize import minimize


class Optimizer(ABC):
    """Optimize our quantum circuit."""

    __slots__ = "__dict__"

    @abstractmethod
    def __call__(self, cost: Callable, grad_cost: Callable, init_params: ndarray) -> ndarray:
        """
        Calculate the optimized parameters.

        Parameters
        ----------
        cost: Callable
            Cost function to be minimized.
        grad_cost: Callable
            Gradient of the cost function.
        init_params : ndarray
            Initial parameter guess for the cost function; used to initialize the optimizer.

        Returns
        -------
        ndarray
            Optimum parameters
        """
        ...


class BlackBoxOptimizer(Optimizer):
    """
    Optimizer that uses scipy's inbuilt function `minimize`.

    Attributes
    ----------
    method: str
        The desired optimization method.
    method_kwargs : dict
        A dictionary with keyword arguments for the optimizer.
    """

    __slots__ = (
        "method",
        "method_kwargs",
    )  # faster memory access to the attributes than using __dict__

    blackbox_methods = ["CG", "L-BFGS-B", "COBYLA", "SLSQP"]

    def __init__(self, method: str, method_kwargs: dict = {}):
        """
        Initialize a black box optimizer.

        Parameters
        ----------
        method : str
            The desired optimization method.
        method_kwargs : dict
            A dictionary with keyword arguments for the optimizer.
        """
        if method in BlackBoxOptimizer.blackbox_methods:
            self.method = method
            self.method_kwargs = method_kwargs
        else:
            raise ValueError(f"Optimization {method} is not supported.")

    def __call__(self, cost: Callable, grad_cost: Callable, init_params: ndarray) -> ndarray:
        """
        Calculate the optimized parameters using `scipy.optimize.minimize()`.

        Parameters
        ----------
        cost: Callable
            Cost function to be minimized.
        grad_cost: Callable
            Gradient of the cost function.
        init_params : ndarray
            Initial parameter guess for the cost function; used to initialize the optimizer.

        Returns
        -------
        ndarray
            Optimum parameters
        """
        result = minimize(
            cost, init_params, method=self.method, jac=grad_cost, options=self.method_kwargs
        )
        params = result.x
        return params


class GDOptimizer(Optimizer):
    """Gradient descent optimizer."""

    __slots__ = "step_size", "iter_index", "min_cost"

    def __init__(self, iters: int, step_size: float):
        """
        Parameters
        ----------
        iters : int
            The number of gradient descent iterations to perform.
        step_size : float
            The size of the step of each gradient descent iteration.
        """
        self.step_size = step_size
        self._iters = iters

    @property
    def iters(self):
        """Number of iterations of gradient descent."""
        return self._iters

    @iters.setter
    def iters(self, new_iters):
        """In case we want to update the number of iterations."""
        self._iters = new_iters

    def __call__(self, cost: Callable, grad_cost: Callable, params: ndarray) -> ndarray:
        """
        Calculate the optimized parameters using a number of gradient descent iterations.

        Parameters
        ----------
        cost : Callable
            Cost function to be minimized.
        grad_cost : Callable
            Gradient of the cost function.
        params : ndarray
            Initial parameter guess for the cost function; used to initialize the optimizer.

        Returns
        -------
        ndarray
            Optimum parameters
        """
        min_cost = 100000
        min_params = zeros_like(params)

        for i in range(self.iters):
            self.iter_index = i
            params = self.step(grad_cost, params)

            if (c := cost(params)) < min_cost:
                self.min_cost = c
                min_params = params

        return min_params

    def step(self, grad_cost: Callable, params: ndarray) -> ndarray:
        """Update the parameters with a step of Gradient Descent."""
        params = params - grad_cost(params) * self.step_size
        return params


class AdamOptimizer(GDOptimizer):
    """
    Adam (A Method for Stochastic Optimization) optimizer.

    Attributes
    ----------
    step_size : float
        The size of the step of each gradient descent iteration.
    beta1 : float
        The factor for the average gradient.
    beta2 : float
        The factor for the average squared gradient.
    eps: float
        A regularizing small parameter used to avoid division by zero.

    References
    ----------
    The optimizer is described in [1]_.

    .. [1] https://arxiv.org/abs/1412.6980
    """

    __slots__ = "alpha", "beta1", "beta2", "eps"

    def __init__(
        self,
        iters: int,
        step_size: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        """
        Parameters
        ----------
        iters : int
            The number of gradient descent iterations to perform.
        step_size : float
            The size of the step of each gradient descent iteration.
        beta1 : float
            The factor for the average gradient.
        beta2 : float
            The factor for the average squared gradient.
        eps: float
            A regularizing small parameter used to avoid division by zero.
        """
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        super().__init__(iters, step_size)

    def step(self, grad_cost: Callable, params: ndarray) -> ndarray:
        """Update the parameters with a step of Adam. Adam changes the step size in each iteration."""
        m = zeros_like(params)
        v = zeros_like(params)
        grad = grad_cost(params)

        m = self.beta1 * m + (1.0 - self.beta1) * grad
        v = self.beta2 * v + (1.0 - self.beta2) * grad**2
        mhat = m / (1.0 - self.beta1 ** (self.iter_index + 1))
        vhat = v / (1.0 - self.beta2 ** (self.iter_index + 1))
        params = params - self.step_size * mhat / (sqrt(vhat) + self.eps)

        return params
