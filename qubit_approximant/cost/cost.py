"""Module docstrings."""

import numpy as np

from qubit_approximant.model import Model


def mse(fn: np.ndarray, fn_approx: np.ndarray) -> float:
    fn_diff = fn_approx - fn
    return np.mean(np.absolute(fn_diff) ** 2)


def grad_mse(fn: np.ndarray, fn_approx: np.ndarray, grad_fn_approx: np.ndarray) -> np.ndarray:
    fn_diff = fn_approx - fn
    return 2 * np.real(np.einsum("g, gi -> i", fn_diff.conj(), grad_fn_approx)) / fn.size


def mse_weighted(fn: np.ndarray, fn_approx: np.ndarray) -> float:
    fn_diff = fn_approx - fn
    return np.mean(fn * np.absolute(fn_diff) ** 2)


def grad_mse_weighted(fn: np.ndarray, fn_approx: np.ndarray, grad_fn_approx: np.ndarray) -> np.ndarray:
    fn_diff = fn_approx - fn
    return 2 * np.real(np.einsum("g, g, gi -> i", fn, fn_diff.conj(), grad_fn_approx)) / fn.size  # fn is real!!


def rmse(fn: np.ndarray, fn_approx: np.ndarray) -> float:
    return np.sqrt(mse(fn, fn_approx))


def grad_rmse(fn: np.ndarray, fn_approx: np.ndarray, grad_fn_approx: np.ndarray) -> np.ndarray:
    fn_diff = fn_approx - fn
    coef = 1 / (np.sqrt(fn.size) * np.sqrt(np.sum(np.abs(fn_diff) ** 2) + 1e-9))
    return coef * np.real(np.einsum("g, gi -> i", fn_diff.conj(), grad_fn_approx))


class Cost:
    '''Create a cost function from the encoding and the metric.'''

    def __init__(self, fn: np.ndarray, model: Model, metric: str):
        try:
            self.metric = globals()[metric]
            self.grad_metric = globals()["grad_" + metric]
        except KeyError:
            raise ValueError("Invalid metric '{metric}'. Choose between 'MSE' or 'RMSE'.")

        self.model = model
        self.fn = fn

    def __call__(self, params) -> float:
        fn_approx = self.model(params)
        return self.metric(self.fn, fn_approx)

    def grad(self, params) -> np.ndarray:
        grad_fn_approx, fn_approx = self.model.grad(params)
        return self.grad_metric(self.fn, fn_approx, grad_fn_approx)
