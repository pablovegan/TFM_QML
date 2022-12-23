"""
Module docstrings
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

from numpy import ndarray, zeros_like, sqrt
from autograd import grad
from scipy.optimize import minimize

from qnl_library.circuit import Circuit
from .cost_function import CostFunction


class Optimizer(ABC):
    """Optimize our quantum circuit."""

    __slots__ = ()

    @abstractmethod
    def __call__(self, init_params: ndarray, **method_kwargs):
        ...
        

class BlackBoxOptimizer(Optimizer):
    """
    Optimizer that uses scipy's inbuilt function `minimize`.
    
    Attributes
    ----------
    method: str
        The desired optimization method.
    """

    __slots__ = 'method', '__dict__'  # faster memory access to the attributes than using __dict__

    blackbox_methods = ['CG', 'L-BFGS-B', 'COBYLA', 'SLSQP']

    def __init__(self, method: str):
        """
        Initialize a black box optimizer.
        
        Parameters
        ----------
        method: str
            The desired optimization method.
        """
        if method in BlackBoxOptimizer.blackbox_methods:
            self.method = method 
        else:
            raise ValueError(f"Optimization {method} is not supported.") 

    def __call__(self, cost_fn: Callable, init_params: ndarray, **method_kwargs) -> ndarray:    
        """
        Calculate the optimized parameters using `scipy.optimize.minimize()`.
        
        Parameters
        ----------
        cost_fn: Callable
            Cost function to be minimized.
        init_params : ndarray
            Initial parameter guess for the cost function; used to initialize the optimizer.
        **kwargs
            Specific keyword arguments for the chosen optimizer.
            The keyword arguments are passed to `scipy.optimize.minimize().` 
        """
        result = minimize(cost_fn, init_params, method = self.method, **method_kwargs)
        opt_params = result.x
        return opt_params # optimal parameters


class GDOptimizer(Optimizer):
    """Gradient descent optimizer."""

    __slots__ = 'step_size', 'iter_index', '__dict__'

    def __init__(self, step_size: float):

        self.step_size = step_size

    def __call__(self, cost_fn: Callable, init_params: Callable, iters: int) -> ndarray:
        """
        Calculate the optimized parameters using a number of gradient descent iterations.
        
        Parameters
        ----------
        cost_fn: Callable
            Cost function to be minimized.
        init_params: ndarray
            Initial parameter guess for the cost function; used to initialize the optimizer.
        iter: int
            Number of iterations of gradient descent to perform.
        """
        self.iter_index = 0
        params = init_params
        
        for _ in range(iters):
            params = self.step(cost_fn, params)
            self.iter_index += 1
        return params
            
    def step(self, cost_fn: Callable, params: ndarray, grad_fn = None, **kwargs) -> ndarray:
        """
        Update the parameters with a step of Gradient Descent.
        
        Parameters
        ----------
        iter_index: int
            Adam changes the step size in each iteration, so it requires the index of the current iteration. 
        cost_fn: Callable
            The objective function for optimization.
        params: ndarray
            The current value of the parameters to be uptdated.
        grad_fn: Optional[Callable]
            The gradient of the cost function.
        **kwargs
            Extra keyword arguments for the cost function.
        """

        grad_fn = grad(cost_fn, 0) if grad_fn is None else grad_fn
        grad_fn = grad_fn(params, **kwargs)
    
        params = params - grad_fn * self.step_size
        return params


class AdamOptimizer(GDOptimizer):
    """
    Adam (A Method for Stochastic Optimization) optimizer.
    
    Attributes
    ----------
    alpha : float
        steps size
    beta1 : float
        factor for average gradient
    beta2 : float
        factor for average squared gradient
    eps: float
        regularizing small parameter used to avoid division by zero

    References
    ----------
    The optimizer is described in [1]_.

    .. [1] https://arxiv.org/abs/1412.6980
    """

    __slots__ = 'alpha', 'beta1', 'beta2', 'eps'

    def __init__(self,
                alpha: float = 0.01,
                beta1: float = 0.9,
                beta2: float = 0.999,
                eps: float = 1e-8
                ):
        """
        Parameters
        ----------
        alpha : float
            steps size
        beta1 : float
            factor for average gradient
        beta2 : float
            factor for average squared gradient
        eps: float
            regularizing small parameter used to avoid division by zero
        """
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
    
    def step(self, cost_fn: Callable, params: ndarray, grad_fn: Callable, **kwargs) -> ndarray:
        """
        Update the parameters with a step of Adam. Adam changes the step size in each iteration.
        
        Parameters
        ----------
        cost_fn: Callable
            The objective function for optimization.
        params: ndarray
            The current value of the parameters to be uptdated.
        grad_fn: Optional[Callable]
            The gradient of the cost function.
        **kwargs
            Extra keyword arguments for the cost function.
        """
        m = zeros_like(params)
        v = zeros_like(params)
        grad = grad_fn(params, **kwargs)
        
        m = self.beta1 * m + (1.0 - self.beta1) * grad
        v = self.beta2 * v + (1.0 - self.beta2) * grad**2
        mhat = m / (1.0 - self.beta1**(super().iter_index + 1))
        vhat = v / (1.0 - self.beta2**(super().iter_index + 1))
        params = params - self.alpha * mhat / (sqrt(vhat) + self.eps)
        
        return params
