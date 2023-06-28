import pytest
from typing import Type

import numpy as np

from qubit_approximant.core import Circuit, CircuitRxRyRz, CircuitRxRy, Cost
from qubit_approximant.core.optimizer import BlackBoxOptimizer, AdamOptimizer


x = np.linspace(-2, 2, 100)
fn = np.exp(-((x) ** 2) / (2 * 0.5**2)) / (0.5 * np.sqrt(2 * np.pi))
layers = 8
np.random.seed(20)
rng = np.random.default_rng()
params = 0.7 * rng.standard_normal(4 * layers)


@pytest.mark.parametrize(
    ("circuit_class", "encoding", "metric", "params", "tol"),
    (
        (CircuitRxRyRz, "prob", "mse", rng.standard_normal(4 * layers), 1e-4),
        (CircuitRxRyRz, "amp", "mse", rng.standard_normal(4 * layers), 1e-4),
        (CircuitRxRyRz, "prob", "rmse", rng.standard_normal(4 * layers), 5e-3),
        (CircuitRxRyRz, "amp", "rmse", rng.standard_normal(4 * layers), 5e-3),
        (CircuitRxRy, "prob", "mse", rng.standard_normal(3 * layers), 1e-3),
        (CircuitRxRy, "prob", "rmse", rng.standard_normal(3 * layers), 1e-3),
        (CircuitRxRy, "amp", "rmse", rng.standard_normal(3 * layers), 1e-2),
        (CircuitRxRy, "amp", "mse", rng.standard_normal(3 * layers), 1e-2),
    ),
)
def test_blackbox(
    circuit_class: Type[Circuit], encoding: str, metric: str, params: np.ndarray, tol: float
):
    model = circuit_class(x=x, encoding=encoding)  # type: ignore
    cost = Cost(fn, model, metric=metric)
    opt = BlackBoxOptimizer(method="L-BFGS-B")
    params = opt(cost, cost.grad, params)
    assert cost(params) < tol


@pytest.mark.parametrize(
    ("circuit_class", "encoding", "metric", "params", "tol"),
    (
        (CircuitRxRyRz, "prob", "mse", rng.standard_normal(4 * layers), 0.1),
        (CircuitRxRyRz, "prob", "rmse", rng.standard_normal(4 * layers), 0.1),
    ),
)
def test_adam(
    circuit_class: Type[Circuit], encoding: str, metric: str, params: np.ndarray, tol: float
):
    model = circuit_class(x=x, encoding=encoding)  # type: ignore
    cost = Cost(fn, model, metric=metric)
    opt = AdamOptimizer(5000)
    params = opt(cost, cost.grad, params)
    assert cost(params) < tol  # Adam optimizer is not working very good :(
