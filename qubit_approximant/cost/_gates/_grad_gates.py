"""Gates for our quantum circuit."""

import numpy as np
from numpy import cos, sin, ndarray


def grad_RX(angle: float) -> ndarray:
    """Derivative of the rotation around X axis."""
    return 0.5 * np.array(
        [[-sin(angle / 2), -1j * cos(angle / 2)], [-1j * cos(angle / 2), -sin(angle / 2)]]
    )


def grad_RY(angle: float) -> ndarray:
    """Derivative of the rotation around Y axis."""
    return 0.5 * np.array([[-sin(angle / 2), -cos(angle / 2)], [cos(angle / 2), -sin(angle / 2)]])


def grad_RZ(angle: float) -> ndarray:
    """Derivative of the rotation around Z axis."""
    return 0.5 * np.array(
        [[-1j * cos(angle / 2) - sin(angle / 2), 0], [0, 1j * cos(angle / 2) - sin(angle / 2)]]
    )
