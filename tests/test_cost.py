import pytest

import numpy as np
from scipy.optimize import check_grad

from qubit_approximant import Cost, CircuitRxRyRz, CircuitRxRy, CircuitRy


x = np.linspace(-2, 2, 100)
fn = np.exp(-((x) ** 2) / (2 * 0.5**2)) / (0.5 * np.sqrt(2 * np.pi))
rng = np.random.default_rng()
params = 0.3 * rng.standard_normal(4 * 6)


@pytest.mark.parametrize(
    ("x", "fn", "encoding", "metric", "params"),
    (
        (x, fn, "amp", "mse", params),
        (x, fn, "prob", "mse", params),
        (x, fn, "amp", "rmse", params),
        (x, fn, "prob", "rmse", params),
        (x, fn, "amp", "mse_weighted", params),
        (x, fn, "prob", "mse_weighted", params),
        (x, fn, "prob", "log_cosh", params),
        (x, fn, "prob", "kl_divergence", params),
    ),
)
def test_grad_CircuitRxRyRz(
    x: np.ndarray, fn: np.ndarray, encoding: str, metric: str, params: np.ndarray
) -> None:
    circuit = CircuitRxRyRz(x=x, encoding=encoding)
    cost = Cost(fn, circuit, metric=metric)
    assert (g := check_grad(cost, cost.grad, params)) < 1e-5, f"Check_grad = {g}"


params = 0.3 * rng.standard_normal(3 * 6)


@pytest.mark.parametrize(
    ("x", "fn", "encoding", "metric", "params"),
    (
        (x, fn, "amp", "mse", params),
        (x, fn, "prob", "mse", params),
        (x, fn, "amp", "rmse", params),
        (x, fn, "prob", "rmse", params),
        (x, fn, "amp", "mse_weighted", params),
        (x, fn, "prob", "mse_weighted", params),
        (x, fn, "prob", "log_cosh", params),
        (x, fn, "prob", "kl_divergence", params),
    ),
)
def test_grad_CircuitRxRy(
    x: np.ndarray, fn: np.ndarray, encoding: str, metric: str, params: np.ndarray
) -> None:
    circuit = CircuitRxRy(x=x, encoding=encoding)
    cost = Cost(fn, circuit, metric=metric)
    assert (g := check_grad(cost, cost.grad, params)) < 1e-5, f"Check_grad = {g}"


params = 0.3 * rng.standard_normal(2 * 6)


@pytest.mark.parametrize(
    ("x", "fn", "encoding", "metric", "params"),
    (
        (x, fn, "amp", "mse", params),
        (x, fn, "prob", "mse", params),
        (x, fn, "amp", "rmse", params),
        (x, fn, "prob", "rmse", params),
        (x, fn, "amp", "mse_weighted", params),
        (x, fn, "prob", "mse_weighted", params),
        (x, fn, "prob", "log_cosh", params),
        (x, fn, "prob", "kl_divergence", params),
    ),
)
def test_grad_CircuitRy(
    x: np.ndarray, fn: np.ndarray, encoding: str, metric: str, params: np.ndarray
) -> None:
    circuit = CircuitRy(x=x, encoding=encoding)
    cost = Cost(fn, circuit, metric=metric)
    assert (g := check_grad(cost, cost.grad, params)) < 1e-5, f"Check_grad = {g}"
