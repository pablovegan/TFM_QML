"""Metrics and their gradients to use in the cost function and optimization process."""

import numpy as np
from numpy.typing import NDArray


def mse(fn: NDArray, fn_approx: NDArray) -> float:
    """Returns the minimum square error."""
    fn_diff = fn_approx - fn
    return np.mean(np.absolute(fn_diff) ** 2)


def grad_mse(fn: NDArray, fn_approx: NDArray, grad_fn_approx: NDArray) -> NDArray:
    """Returns the gradient of the minimum square error."""
    fn_diff = fn_approx - fn
    return 2 * np.real(np.einsum("g, gi -> i", fn_diff.conj(), grad_fn_approx)) / fn.size


def mse_weighted(fn: NDArray, fn_approx: NDArray) -> float:
    """Returns a weighted minimum square error."""
    fn_diff = fn_approx - fn
    return np.mean(fn * np.absolute(fn_diff) ** 2)


def grad_mse_weighted(fn: NDArray, fn_approx: NDArray, grad_fn_approx: NDArray) -> NDArray:
    """Returns the gradient of the weighted minimum square error."""
    fn_diff = fn_approx - fn
    return (
        2 * np.real(np.einsum("g, g, gi -> i", fn, fn_diff.conj(), grad_fn_approx)) / fn.size
    )  # fn is real!!


def rmse(fn: NDArray, fn_approx: NDArray) -> float:
    """Returns the root minimum square error."""
    return np.sqrt(mse(fn, fn_approx))


def grad_rmse(fn: NDArray, fn_approx: NDArray, grad_fn_approx: NDArray) -> NDArray:
    """Returns the gradient of the root minimum square error."""
    fn_diff = fn_approx - fn
    coef = 1 / (np.sqrt(fn.size) * np.sqrt(np.sum(np.abs(fn_diff) ** 2) + 1e-9))
    return coef * np.real(np.einsum("g, gi -> i", fn_diff.conj(), grad_fn_approx))


def kl_divergence(fn: NDArray, fn_approx: NDArray) -> float:
    """Returns the KL divergence. This metric should be used
    with strictly real positive functions"""
    return np.mean(fn * np.log(fn_approx / fn))


def grad_kl_divergence(fn: NDArray, fn_approx: NDArray, grad_fn_approx: NDArray) -> NDArray:
    """Returns the gradient of the KL divergence."""
    return np.real(np.einsum("g, gi -> i", fn / fn_approx, grad_fn_approx)) / fn.size


def log_cosh(fn: NDArray, fn_approx: NDArray) -> float:
    """Returns the logarithm of the hyperbolic cosine. This metric
    should be used with strictly real positive functions"""
    fn_diff = fn_approx - fn
    return np.mean(np.log(np.cosh(fn_diff)))


def grad_log_cosh(fn: NDArray, fn_approx: NDArray, grad_fn_approx: NDArray) -> NDArray:
    """Returns the gradiend of the logarithm of the hyperbolic cosine."""
    fn_diff = fn_approx - fn
    return np.real(np.einsum("g, gi -> i", np.tanh(fn_diff), grad_fn_approx)) / fn.size
