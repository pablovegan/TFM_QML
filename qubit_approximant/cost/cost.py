"""Module docstrings."""

import numpy as np

from ..cost import Model, Metric


class Cost:
    """Create a cost function from the encoding and the metric."""

    def __init__(self, fn: np.ndarray, model: Model, metric: Metric):
        self.model = model
        self.metric = metric
        self.fn = fn

    def __call__(self, params) -> float:
        w, θ = Cost.split(params)
        fn_apprx = self.model.encoding(θ, w)
        return self.metric(fn_apprx - self.fn)

    def grad(self, params) -> np.ndarray:
        w, θ = Cost.split(params)
        grad_fn, fn_approx = self.model.grad_encoding(θ, w)
        return self.metric.grad(fn_approx - self.fn, grad_fn)

    @staticmethod
    def split(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        layers = params.size // 4
        return params[0:layers], params[layers:].reshape(3, layers)
