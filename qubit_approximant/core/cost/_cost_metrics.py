"""Metrics and their gradients to use in the cost function and optimization process."""

import numpy as np


def mse(fn: np.ndarray, fn_approx: np.ndarray) -> float:
    fn_diff = fn_approx - fn
    return np.mean(np.absolute(fn_diff) ** 2)


def grad_mse(fn: np.ndarray, fn_approx: np.ndarray, grad_fn_approx: np.ndarray) -> np.ndarray:
    fn_diff = fn_approx - fn
    return 2 * np.real(np.einsum("g, gi -> i", fn_diff.conj(), grad_fn_approx)) / fn.size


def mse_weighted(fn: np.ndarray, fn_approx: np.ndarray) -> float:
    fn_diff = fn_approx - fn
    return np.mean(fn * np.absolute(fn_diff) ** 2)


def grad_mse_weighted(
    fn: np.ndarray, fn_approx: np.ndarray, grad_fn_approx: np.ndarray
) -> np.ndarray:
    fn_diff = fn_approx - fn
    return (
        2 * np.real(np.einsum("g, g, gi -> i", fn, fn_diff.conj(), grad_fn_approx)) / fn.size
    )  # fn is real!!


def rmse(fn: np.ndarray, fn_approx: np.ndarray) -> float:
    return np.sqrt(mse(fn, fn_approx))


def grad_rmse(fn: np.ndarray, fn_approx: np.ndarray, grad_fn_approx: np.ndarray) -> np.ndarray:
    fn_diff = fn_approx - fn
    coef = 1 / (np.sqrt(fn.size) * np.sqrt(np.sum(np.abs(fn_diff) ** 2) + 1e-9))
    return coef * np.real(np.einsum("g, gi -> i", fn_diff.conj(), grad_fn_approx))


def kl_divergence(fn: np.ndarray, fn_approx: np.ndarray) -> float:
    return np.mean(fn * np.log(fn_approx / fn))


def grad_kl_divergence(
    fn: np.ndarray, fn_approx: np.ndarray, grad_fn_approx: np.ndarray
) -> np.ndarray:
    return np.real(np.einsum("g, gi -> i", fn / fn_approx, grad_fn_approx)) / fn.size


def log_cosh(fn: np.ndarray, fn_approx: np.ndarray) -> float:
    fn_diff = fn_approx - fn
    return np.mean(np.log(np.cosh(fn_diff)))


def grad_log_cosh(fn: np.ndarray, fn_approx: np.ndarray, grad_fn_approx: np.ndarray) -> np.ndarray:
    fn_diff = fn_approx - fn
    return np.real(np.einsum("g, gi -> i", np.tanh(fn_diff), grad_fn_approx)) / fn.size
