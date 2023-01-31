"""
Module docstrings
"""

import numpy as np
from numpy import cos, sin


class Cost:
    def __init__(self, x: np.ndarray, fn: np.ndarray, encoding: str):
        """
        Parameters
        ----------
        layers: int
            The number of layers in our circuit.
        encoding: str
            Choose between amplitude or probability encoding.
        """
        self.x = x
        self.x_size = x.size
        self.fn = fn
        if encoding == "prob":
            self.prob = True  # TODO: quitar esto
            self.encoding = self._prob_encoding
            self.grad_encoding = self._grad_prob_encoding
        elif encoding == "amp":
            self.prob = False
            self.encoding = self._amp_encoding
            self.grad_encoding = self._grad_amp_encoding
        else:
            raise ValueError("Invalid encoding '{encoding}'. Choose between 'prob' or 'amp'.")

    def __call__(self, θ: np.ndarray, w: np.ndarray):
        ...

    def _layer(self, θ: np.ndarray, w: float) -> tuple:
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

    def _amp_encoding(self, θ: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Returns our variational ansatz, the product of the L layers.
        Since we are interested in the amplitude/probability of the |0> qubit
        we select the (0,0) element of the unitary matrix U (for every x).
        """
        U = self._layer(θ[:, 0], w[0])[:, :, 0]
        for i in range(1, w.size):
            Ui = self._layer(θ[:, i], w[i])
            U = np.einsum("gmn, gn -> gm", Ui, U)
        return U[:, 0]

    def _prob_encoding(self, θ: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Returns our variational ansatz, the product of the L layers.
        Since we are interested in the amplitude/probability of the |0> qubit
        we select the (0,0) element of the unitary matrix U (for every x).
        """
        fn_amp = self._amp_encoding(θ, w)
        return fn_amp.real**2 + fn_amp.imag**2

    @staticmethod
    def mse_error(fn_exact: np.ndarray, fn_approx: np.ndarray) -> float:
        return np.mean(np.absolute(fn_exact - fn_approx) ** 2)

    @staticmethod
    def rmse_error(fn_exact: np.ndarray, fn_approx: np.ndarray) -> float:
        return np.sqrt(Cost.mse_error(fn_exact, fn_approx))

    def _der_layer(self, θ: np.ndarray, w: float) -> tuple:
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

        return np.array([Dw, Dx, Dy, Dz])

    def _grad_amp_encoding(self, θ: np.ndarray, w: np.ndarray):
        """ "Create recursively the derivatives with respect to each parameter of the entire net."""

        layers = w.size
        U = np.tensordot(np.ones(self.x_size), np.identity(2), axes=0)  # dim (G,2,2)
        D = np.zeros((layers, 4, self.x_size, 2, 2), dtype=np.complex128)

        for i in range(layers):
            DUi = self._der_layer(θ[:, i], w[i])  # dim (4,G,2,2)
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
        # D is shape (layers,4,x.size)
        D = D[:, :, :, 0, 0].swapaxes(0, 2)  # D is shape (x.size, 4, layers)
        # return (D.reshape(self.x_size, -1), U[:, 0, 0])
        return D.reshape(self.x_size, -1), U[:, 0, 0]  # D has shape (x, L*4)

    def _grad_prob_encoding(self, θ: np.ndarray, w: np.ndarray):

        grad_amp, amp_enc = self._grad_amp_encoding(θ, w)
        fn_approx = amp_enc.real**2 + amp_enc.imag**2
        return 2 * np.real(np.einsum("g, gi -> gi", amp_enc.conj(), grad_amp)), fn_approx

    def grad_mse(self, θ: np.ndarray, w: np.ndarray):

        grad, fn_approx = self.grad_encoding(θ, w)
        fn_diff = fn_approx - self.fn

        return 2 * np.real(np.einsum("g, gi -> i", fn_diff.conj(), grad)) / self.x_size

    def grad_rmse(self, θ: np.ndarray, w: np.ndarray):

        grad, fn_approx = self.grad_encoding(θ, w)
        fn_diff = fn_approx - self.fn
        coef = 1 / (np.sqrt(self.x_size) * np.sqrt(np.sum(np.abs(fn_diff) ** 2) + 1e-9))
        return coef * np.real(np.einsum("g, gi -> i", fn_diff.conj(), grad))
