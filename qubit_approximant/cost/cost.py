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
        fn_apprx = self.model(params)
        return self.metric(fn_apprx - self.fn)

    def grad(self, params) -> np.ndarray:
        grad_fn, fn_approx = self.model.grad(params)
        return self.metric.grad(fn_approx - self.fn, grad_fn)
