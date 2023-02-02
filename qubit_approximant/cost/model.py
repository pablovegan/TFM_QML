"""
Module docstrings
"""

import numpy as np
from numpy import cos, sin, ndarray


def split(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the parameters into"""
    assert params.size % 4 == 0, "Error: number of parameters must equal 4 * layers."
    layers = params.size // 4
    return params[0:layers], params[layers:].reshape(3, layers)


class Model:
    """
    Quantum circuit that encodes our function. The circuit consists of
    a number of layers,

    U = Ln * ... * L1

    each of which is made of three rotations dependent
    on four parameters:

    L = RX(x * w + θ0) RY(θ1) RZ(θ2)
    """

    __slots__ = "x", "encoding", "grad"

    def __init__(self, x: ndarray, encoding: str):
        """
        Parameters
        ----------
        x: ndarray
            The values where we wish to approximate a function.
        encoding: str
            Choose between amplitude or probability encoding.
            Must be either 'amp' or 'prob'.
        """
        self.x = x
        if encoding == "prob":
            self.encoding = self._prob_encoding
            self.grad = self._grad_prob
        elif encoding == "amp":
            self.encoding = self._amp_encoding
            self.grad = self._grad_amp
        else:
            raise ValueError("Invalid encoding '{encoding}'. Choose between 'prob' or 'amp'.")

    def __call__(self, params: ndarray):
        """
        Each layer is the product of three rotations.

        Parmeters
        ---------
        θ : (3, layers) ndarray
            Bias parameters of each rotation.
        w : (layers) ndarray
            Weights of the RsX rotation.

        Returns
        -------
        (x.size) ndarray
            Values of the function encoded in our qubit.
        """
        w, θ = split(params)
        return self.encoding(θ, w)

    def _layer(self, θ: ndarray, w: float) -> ndarray:
        """
        Each layer is the product of three rotations.

        Parmeters
        ---------
        θ : (3) array
            Bias parameters of each rotation.
        w : float
            Weight of the X rotation.

        Returns
        -------
        A : (G,2,2) array
            Unitary matrix of the layer.
        """
        ϕ = w * self.x + θ[0]
        Rx = np.array([[cos(ϕ / 2), -1j * sin(ϕ / 2)], [-1j * sin(ϕ / 2), cos(ϕ / 2)]])
        Ry = np.array([[cos(θ[1] / 2), -sin(θ[1] / 2)], [sin(θ[1] / 2), cos(θ[1] / 2)]])
        Rz = np.array(
            [[cos(θ[2] / 2) - 1j * sin(θ[2] / 2), 0], [0, cos(θ[2] / 2) + 1j * sin(θ[2] / 2)]]
        )
        # move the x axis to first position
        return np.einsum("mn, np, pqg -> gmq", Rz, Ry, Rx)

    def _amp_encoding(self, θ: ndarray, w: ndarray) -> ndarray:
        """Returns approximate function encoded in the amplitude of the qubit."""
        U = self._layer(θ[:, 0], w[0])[:, :, 0]
        for i in range(1, w.size):
            Ui = self._layer(θ[:, i], w[i])
            U = np.einsum("gmn, gn -> gm", Ui, U)
        return U[:, 0]

    def _prob_encoding(self, θ: ndarray, w: ndarray) -> ndarray:
        """Returns approximate function encoded in the probability of the qubit."""
        fn_amp = self._amp_encoding(θ, w)
        return fn_amp.real**2 + fn_amp.imag**2

    def _grad_layer(self, θ: ndarray, w: float) -> ndarray:
        """Returns the derivative of one layer with respect to its 4 parameters."""
        ϕ = w * self.x + θ[0]

        Rx = np.array([[cos(ϕ / 2), -1j * sin(ϕ / 2)], [-1j * sin(ϕ / 2), cos(ϕ / 2)]])
        Ry = np.array([[cos(θ[1] / 2), -sin(θ[1] / 2)], [sin(θ[1] / 2), cos(θ[1] / 2)]])
        Rz = np.array(
            [[cos(θ[2] / 2) - 1j * sin(θ[2] / 2), 0], [0, cos(θ[2] / 2) + 1j * sin(θ[2] / 2)]]
        )

        DRx = 0.5 * np.asarray([[-sin(ϕ / 2), -1j * cos(ϕ / 2)], [-1j * cos(ϕ / 2), -sin(ϕ / 2)]])
        DRy = 0.5 * np.array([[-sin(θ[1] / 2), -cos(θ[1] / 2)], [cos(θ[1] / 2), -sin(θ[1] / 2)]])
        DRz = 0.5 * np.array(
            [[-1j * cos(θ[2] / 2) - sin(θ[2] / 2), 0], [0, 1j * cos(θ[2] / 2) - sin(θ[2] / 2)]]
        )

        Dx = np.einsum("mn, np, pqg -> gmq", Rz, Ry, DRx)
        Dw = np.einsum("gmq, g -> gmq", Dx, self.x)
        Dy = np.einsum("mn, np, pqg -> gmq", Rz, DRy, Rx)
        Dz = np.einsum("mn, np, pqg -> gmq", DRz, Ry, Rx)

        return np.array([Dw, Dx, Dy, Dz])  # type: ignore

    def _grad_amp(self, params: ndarray) -> tuple[ndarray, ndarray]:
        """Returns the gradient of the amplitude encoding and the encoded function."""
        w, θ = split(params)
        layers = w.size
        U = np.tensordot(np.ones(self.x.size), np.identity(2), axes=0)  # dim (G,2,2)
        D = np.zeros((layers, 4, self.x.size, 2, 2), dtype=np.complex128)

        for i in range(layers):
            DUi = self._grad_layer(θ[:, i], w[i])  # dim (4,G,2,2)
            # j is each of the derivatives
            D[i, ...] = np.einsum("jgmn, gnp -> jgmp", DUi, U)
            # Multiply derivative times next layer
            Ui = self._layer(θ[:, i], w[i])
            U = np.einsum("gmn, gnp -> gmp", Ui, U)
        # In the first iteration we reuse the L-th layer
        B = Ui
        for i in range(layers - 2, -1, -1):
            D[i, ...] = np.einsum("gmn, jgnp -> jgmp", B, D[i, ...])
            # Multiply derivative times previous layer
            Ui = self._layer(θ[:, i], w[i])
            B = np.einsum("gin, gnj -> gij", B, Ui)

        D = D[:, :, :, 0, 0]  # D is shape (layers,4,x.size)
        D = D.swapaxes(0, 2)  # D is shape (x.size, 4, layers)
        grad = D.reshape(self.x.size, -1)  # D has shape (x, L*4)
        fn_approx = U[:, 0, 0]

        return grad, fn_approx

    def _grad_prob(self, params: ndarray) -> tuple[ndarray, ndarray]:
        """Returns the gradient of the probability encoding and the encoded function."""
        grad_amp, amp = self._grad_amp(params)
        fn_approx = amp.real**2 + amp.imag**2
        grad_prob = 2 * np.real(np.einsum("g, gi -> gi", amp.conj(), grad_amp))
        return grad_prob, fn_approx
