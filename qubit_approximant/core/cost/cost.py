"""Module docstrings."""

from numpy import ndarray

from qubit_approximant.model import Model
from ._cost_metrics import *  # noqa


class Cost:
    """Create a cost function from the encoding and the metric."""

    def __init__(self, fn: ndarray, model: Model, metric: str):
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

    def grad(self, params) -> ndarray:
        grad_fn_approx, fn_approx = self.model.grad(params)
        return self.grad_metric(self.fn, fn_approx, grad_fn_approx)
