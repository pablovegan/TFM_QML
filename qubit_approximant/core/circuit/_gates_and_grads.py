"""Gates for our quantum circuit."""

import numpy as np
from numpy import cos, sin
from numpy.typing import NDArray


def RX(angle: float) -> NDArray:
    """Rotation around X axis."""
    return np.array([[cos(angle / 2), -1j * sin(angle / 2)], [-1j * sin(angle / 2), cos(angle / 2)]])


def RY(angle: float) -> NDArray:
    """Rotation around Y axis."""
    return np.array([[cos(angle / 2), -sin(angle / 2)], [sin(angle / 2), cos(angle / 2)]])


def RZ(angle: float) -> NDArray:
    """Rotation around Z axis."""
    return np.array([[cos(angle / 2) - 1j * sin(angle / 2), 0], [0, cos(angle / 2) + 1j * sin(angle / 2)]])


def grad_RX(angle: float) -> NDArray:
    """Derivative of the rotation around X axis."""
    return 0.5 * np.array([[-sin(angle / 2), -1j * cos(angle / 2)], [-1j * cos(angle / 2), -sin(angle / 2)]])


def grad_RY(angle: float) -> NDArray:
    """Derivative of the rotation around Y axis."""
    return 0.5 * np.array([[-sin(angle / 2), -cos(angle / 2)], [cos(angle / 2), -sin(angle / 2)]])


def grad_RZ(angle: float) -> NDArray:
    """Derivative of the rotation around Z axis."""
    return 0.5 * np.array([[-1j * cos(angle / 2) - sin(angle / 2), 0], [0, 1j * cos(angle / 2) - sin(angle / 2)]])
