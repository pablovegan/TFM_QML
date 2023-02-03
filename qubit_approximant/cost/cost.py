"""Module docstrings."""

import numpy as np

from qubit_approximant.model import Model


def mse(fn: np.ndarray) -> float:
    return np.mean(np.absolute(fn) ** 2)


def grad_mse(fn: np.ndarray, grad_fn: np.ndarray) -> np.ndarray:
    return 2 * np.real(np.einsum("g, gi -> i", fn.conj(), grad_fn)) / fn.size


def rmse(fn: np.ndarray) -> float:
    return np.sqrt(mse(fn))


def grad_rmse(fn: np.ndarray, grad_fn: np.ndarray) -> np.ndarray:
    coef = 1 / (np.sqrt(fn.size) * np.sqrt(np.sum(np.abs(fn) ** 2) + 1e-9))
    return coef * np.real(np.einsum("g, gi -> i", fn.conj(), grad_fn))


class Cost:
    """Create a cost function from the encoding and the metric."""

    def __init__(self, fn: np.ndarray, model: Model, metric: str):
        try:
            self.metric = globals()[metric]
            self.grad_metric = globals()["grad_" + metric]
        except KeyError:
            raise ValueError("Invalid metric '{metric}'. Choose between 'MSE' or 'RMSE'.")

        self.model = model
        self.fn = fn

    def __call__(self, params) -> float:
        fn_apprx = self.model(params)
        return self.metric(fn_apprx - self.fn)

    def grad(self, params) -> np.ndarray:
        grad_fn, fn_approx = self.model.grad(params)
        return self.grad_metric(fn_approx - self.fn, grad_fn)
