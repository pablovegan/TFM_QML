from typing import Callable

import numpy as np
import scipy.integrate as integrate


def l1_norm(fn: Callable, fn_approx: Callable, x_limits: tuple[float, float]) -> float:
    """L1 norm of the difference of the given functions 'fn' and 'fn_approx'
     in the interval given by (x_limits[0], x_limits[1]).

    Parameters
    ----------
    fn : Callable
        Function we want to approximate.
    fn_approx : Callable
        Approximation to the function.
    x_limits : tuple[float, float]
        Limits of the integration interval to estimate the L1 norm of the difference
        between the function and the approximation.

    Returns
    -------
    float
        Value of the L1 norm (it is an integral).
    """
    diff_l1 = lambda x: np.abs(fn(x) - fn_approx(x))  # noqa
    return integrate.quad(diff_l1, x_limits[0], x_limits[1], limit=300)[0]


def l2_norm(fn: Callable, fn_approx: Callable, x_limits: tuple[float, float]) -> float:
    diff_l2 = lambda x: np.abs(fn(x) - fn_approx(x)) ** 2  # noqa
    return np.sqrt(integrate.quad(diff_l2, x_limits[0], x_limits[1], limit=300)[0])


def inf_norm(fn: Callable, fn_approx: Callable, x_limits: tuple[float, float]) -> float:
    x = np.linspace(x_limits[0], x_limits[1], 10000)
    return np.max(np.abs(fn(x) - fn_approx(x)))


def infidelity(fn: Callable, fn_approx: Callable, x_limits: tuple[float, float]) -> float:
    product_real = lambda x: np.real(fn(x) * fn_approx(x))  # noqa
    product_imag = lambda x: np.imag(fn(x) * fn_approx(x))  # noqa
    fn2 = lambda x: fn(x) ** 2  # noqa
    fn_approx2 = lambda x: np.abs(fn_approx(x)) ** 2  # noqa

    fidelity = (
        integrate.quad(product_real, x_limits[0], x_limits[1], limit=300)[0] ** 2
        + integrate.quad(product_imag, x_limits[0], x_limits[1], limit=300)[0] ** 2
    ) / (
        integrate.quad(fn_approx2, x_limits[0], x_limits[1], limit=300)[0]
        * integrate.quad(fn2, x_limits[0], x_limits[1], limit=300)[0]
    )

    return 1 - fidelity
