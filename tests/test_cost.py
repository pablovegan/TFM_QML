import pytest

import numpy as np
from scipy.optimize import check_grad

from qubit_approximant import Cost, CircuitRxRyRz, CircuitRxRy, CircuitRy


x = np.linspace(-2, 2, 100)
fn = np.exp(-((x) ** 2) / (2 * 0.5**2)) / (0.5 * np.sqrt(2 * np.pi))
params = 0.3 * np.random.randn(4 * 6)


@pytest.mark.parametrize(
    ("x", "fn", "encoding_str", "metric_str", "params"),
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
    x: np.ndarray, fn: np.ndarray, encoding_str: str, metric_str: str, params: np.ndarray
) -> None:
    circuit = CircuitRxRyRz(x=x, encoding_str=encoding_str)
    cost = Cost(fn, circuit, metric_str=metric_str)
    assert (g := check_grad(cost, cost.grad, params)) < 1e-5, f"Check_grad = {g}"


params = 0.3 * np.random.randn(3 * 6)


@pytest.mark.parametrize(
    ("x", "fn", "encoding_str", "metric_str", "params"),
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
    x: np.ndarray, fn: np.ndarray, encoding_str: str, metric_str: str, params: np.ndarray
) -> None:
    circuit = CircuitRxRy(x=x, encoding_str=encoding_str)
    cost = Cost(fn, circuit, metric_str=metric_str)
    assert (g := check_grad(cost, cost.grad, params)) < 1e-5, f"Check_grad = {g}"


params = 0.3 * np.random.randn(2 * 6)


@pytest.mark.parametrize(
    ("x", "fn", "encoding_str", "metric_str", "params"),
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
    x: np.ndarray, fn: np.ndarray, encoding_str: str, metric_str: str, params: np.ndarray
) -> None:
    circuit = CircuitRy(x=x, encoding_str=encoding_str)
    cost = Cost(fn, circuit, metric_str=metric_str)
    assert (g := check_grad(cost, cost.grad, params)) < 1e-5, f"Check_grad = {g}"
