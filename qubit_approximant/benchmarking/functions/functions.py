"""Functions to test our quantum approximator."""

from typing import Optional

import numpy as np
from numpy import ndarray


def gaussian(
    x: ndarray, mean: float = 0.0, std: float = 1, coef: Optional[float] = None
) -> ndarray:
    """Return a gaussian function.

    Parameters
    ----------
    x : ndarray
        Grid in which to approximate the function.
    mean : float, optional
        Mean of the gaussian, by default 0.0
    std : float, optional
        Standard deviation, by default 1
    coef : float, optional
        Factor that multiplies the gaussian., by default None

    Returns
    -------
    ndarray
        Values of the gaussian at each point.
    """
    if coef is None:
        coef = 1 / (std * np.sqrt(2 * np.pi))
    return coef * np.exp(-((x - mean) ** 2) / (2 * std**2))


def lorentzian(x: ndarray, x0: float = 0.0, gamma: float = 1.0) -> ndarray:
    """Return a lorentzian function.

    Parameters
    ----------
    x : ndarray
        Grid in which to approximate the function.
    x0 : float, optional
        Shift x by this value, by default 0.0
    gamma : float, optional
        Parameter of the lorenztian, by default 1.0

    Returns
    -------
    ndarray
        Values of the lorentzian at each point.
    """
    return 1 / np.pi * gamma / ((x - x0) ** 2 + gamma**2)


def sine(x: ndarray, a: float = 1.0, b: float = 0.0) -> ndarray:
    """Return a sine function.

    Parameters
    ----------
    x : ndarray
        Grid in which to approximate the function.
    a : float, optional
        Weight of x in the sine, by default 1.0
    b : float, optional
        Shift of x in the sine, by default 0.0

    Returns
    -------
    ndarray
        Values of the sine at each point.
    """
    return np.sin(a * x + b)


def step(x: ndarray, b: float = 0.0, coef: float = 1.0) -> ndarray:
    """Return a step function.

    Parameters
    ----------
    x : ndarray
        Grid in which to approximate the function.
    b : float, optional
        Shift of x, by default 0.0
    coef : float, optional
        Size of the step, by default 1.0

    Returns
    -------
    ndarray
        Values of the step function at each point.
    """
    return coef * np.heaviside(x, b)


def relu(x: ndarray, a: float = 1.0) -> ndarray:
    r"""Return a relu function
        $$f(x) = \max(0, a \cdot x)$$

    Parameters
    ----------
    x : ndarray
        Grid in which to approximate the function.
    a : float, optional
        Weight of x, by default 1.0

    Returns
    -------
    ndarray
        Values of the relu function at each point.

    Raises
    ------
    ValueError
        "a must be a positive constant"
    """
    if a <= 0:
        raise ValueError("a must be a positive constant")
    return np.maximum(0, a * x)


def tanh(x: ndarray, a: float = 5.0, coef=1.0) -> ndarray:
    """Return a hyperbolic tangent

    Parameters
    ----------
    x : ndarray
        Grid in which to approximate the function.
    a : float, optional
        Weight of x, by default 5.0
    coef : float, optional
        Coefficient of the hyperbolic tangent, by default 1.0

    Returns
    -------
    ndarray
        Values of the relu function at each point.
    """
    return coef * np.tanh(a * x)


def poly(x: ndarray) -> ndarray:
    """Return 4th order a polynomial

    Parameters
    ----------
    x : ndarray
        Grid in which to approximate the function.

    Returns
    -------
    ndarray
        Values of the relu function at each point.
    """
    return np.abs((1 - x**4) * 3 * x**3)


def cos2_sin2(x: ndarray, a: float = 1.0, b: float = 0.0) -> ndarray:
    return np.cos(a * x + b) ** 2 - np.sin(a * x + b) ** 2
