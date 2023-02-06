"""Functions to test our quantum approximator."""

import numpy as np


def gaussian(x: np.ndarray, mean: float = 0.0, std: float = 1, coef=None):
    """
    Approximate and plot a gaussian function.

    Parameters
    ----------
    x : np.ndarray
        Grid in which to approximate the function.
    mean : float
        Mean of the gaussian.
    std : float
        Standard deviation.
    coef : float
        Factor that multiplies the gaussian.

    """
    if coef is None:
        coef = (1 / (std * np.sqrt(2 * np.pi)))
    return coef * np.exp(- (x - mean) ** 2 / (2 * std ** 2))


def lorentzian(x: np.ndarray, x0: float = 0.0, gamma: float = 1.0):
    return 1 / np.pi * gamma / ((x - x0) ** 2 + gamma ** 2)


def sine(x: np.ndarray, a: float = 1.0, b: float = 0.0):
    return np.sin(a*x+b)


def step(x: np.ndarray, b: float = 0.0, coef: float = 1.0):
    return coef*np.heaviside(x, b)


def relu(x: np.ndarray, a: float = 1.0):
    if a <= 0:
        raise ValueError('a must be a positive constant')
    return np.maximum(0, a * x)


def tanh(x: np.ndarray, a: float = 5.0, coef=1.0):
    return coef * np.tanh(a * x)


def poly(x: np.ndarray):
    return np.abs((1 - x ** 4) * 3 * x ** 3)


def cos2_sin2(x: np.ndarray, a: float = 1.0, b: float = 0.0):
    return np.cos(a * x + b) ** 2 - np.sin(a * x + b) ** 2
