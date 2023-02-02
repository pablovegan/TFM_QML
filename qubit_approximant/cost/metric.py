"""Metric for the performance of our model."""

import numpy as np


class Metric:
    def __init__(self, metric: str):
        if metric == "mse":
            self.metric = Metric.mse
            self.grad = Metric.grad_mse
        elif metric == "rmse":
            self.metric = Metric.rmse
            self.grad = Metric.grad_rmse
        else:
            raise ValueError("Invalid metric '{metric}'. Choose between 'MSE' or 'RMSE'.")

    def __call__(self, fn: np.ndarray):
        return self.metric(fn)

    @staticmethod
    def mse(fn: np.ndarray) -> float:
        return np.mean(np.absolute(fn) ** 2)

    @staticmethod
    def rmse(fn: np.ndarray) -> float:
        return np.sqrt(Metric.mse(fn))

    @staticmethod
    def grad_mse(fn: np.ndarray, grad_fn: np.ndarray) -> np.ndarray:
        return 2 * np.real(np.einsum("g, gi -> i", fn.conj(), grad_fn)) / fn.size

    @staticmethod
    def grad_rmse(fn: np.ndarray, grad_fn: np.ndarray) -> np.ndarray:
        coef = 1 / (np.sqrt(fn.size) * np.sqrt(np.sum(np.abs(fn) ** 2) + 1e-9))
        return coef * np.real(np.einsum("g, gi -> i", fn.conj(), grad_fn))
