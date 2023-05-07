"""Functions to test our quantum approximator."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def gaussian(
    x: NDArray, mean: float = 0.0, std: float = 1, coef: Optional[float] = None
) -> NDArray:
    """Return a gaussian function.

    Parameters
    ----------
    x : NDArray
        Grid in which to approximate the function.
    mean : float, optional
        Mean of the gaussian, by default 0.0
    std : float, optional
        Standard deviation, by default 1
    coef : float, optional
        Factor that multiplies the gaussian., by default None

    Returns
    -------
    NDArray
        Values of the gaussian at each point.
    """
    if coef is None:
        coef = 1 / (std * np.sqrt(2 * np.pi))
    return coef * np.exp(-((x - mean) ** 2) / (2 * std**2))


def lorentzian(x: NDArray, x0: float = 0.0, gamma: float = 1.0) -> NDArray:
    """Return a lorentzian function.

    Parameters
    ----------
    x : NDArray
        Grid in which to approximate the function.
    x0 : float, optional
        Shift x by this value, by default 0.0
    gamma : float, optional
        Parameter of the lorenztian, by default 1.0

    Returns
    -------
    NDArray
        Values of the lorentzian at each point.
    """
    return 1 / np.pi * gamma / ((x - x0) ** 2 + gamma**2)


def sine(x: NDArray, a: float = 1.0, b: float = 0.0) -> NDArray:
    """Return a sine function.

    Parameters
    ----------
    x : NDArray
        Grid in which to approximate the function.
    a : float, optional
        Weight of x in the sine, by default 1.0
    b : float, optional
        Shift of x in the sine, by default 0.0

    Returns
    -------
    NDArray
        Values of the sine at each point.
    """
    return np.sin(a * x + b)


def step(x: NDArray, b: float = 0.0, coef: float = 1.0) -> NDArray:
    """Return a step function.

    Parameters
    ----------
    x : NDArray
        Grid in which to approximate the function.
    b : float, optional
        Shift of x, by default 0.0
    coef : float, optional
        Size of the step, by default 1.0

    Returns
    -------
    NDArray
        Values of the step function at each point.
    """
    return coef * np.heaviside(x, b)


def relu(x: NDArray, a: float = 1.0) -> NDArray:
    r"""Return a relu function
        $$f(x) = \max(0, a \cdot x)$$

    Parameters
    ----------
    x : NDArray
        Grid in which to approximate the function.
    a : float, optional
        Weight of x, by default 1.0

    Returns
    -------
    NDArray
        Values of the relu function at each point.

    Raises
    ------
    ValueError
        "a must be a positive constant"
    """
    if a <= 0:
        raise ValueError("a must be a positive constant")
    return np.maximum(0, a * x)


def tanh(x: NDArray, a: float = 5.0, coef=1.0) -> NDArray:
    """Return a hyperbolic tangent

    Parameters
    ----------
    x : NDArray
        Grid in which to approximate the function.
    a : float, optional
        Weight of x, by default 5.0
    coef : float, optional
        Coefficient of the hyperbolic tangent, by default 1.0

    Returns
    -------
    NDArray
        Values of the relu function at each point.
    """
    return coef * np.tanh(a * x)


def poly(x: NDArray) -> NDArray:
    """Return 4th order a polynomial

    Parameters
    ----------
    x : NDArray
        Grid in which to approximate the function.

    Returns
    -------
    NDArray
        Values of the relu function at each point.
    """
    return np.abs((1 - x**4) * 3 * x**3)


def cos2_sin2(x: NDArray, a: float = 1.0, b: float = 0.0) -> NDArray:
    return np.cos(a * x + b) ** 2 - np.sin(a * x + b) ** 2
