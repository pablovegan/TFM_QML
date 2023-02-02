"""Gates for our quantum circuit."""

import numpy as np
from numpy import cos, sin, ndarray


def RX(angle: float) -> ndarray:
    """Rotation around X axis."""
    return np.array(
        [[cos(angle / 2), -1j * sin(angle / 2)], [-1j * sin(angle / 2), cos(angle / 2)]]
    )


def RY(angle: float) -> ndarray:
    """Rotation around Y axis."""
    return np.array([[cos(angle / 2), -sin(angle / 2)], [sin(angle / 2), cos(angle / 2)]])


def RZ(angle: float) -> ndarray:
    """Rotation around Z axis."""
    return np.array(
        [[cos(angle / 2) - 1j * sin(angle / 2), 0], [0, cos(angle / 2) + 1j * sin(angle / 2)]]
    )
