"""
Module docstrings

Classes
-------
Model: abstract class, it is the basis of our specific circuit classes.
RotationsModel: each layer is composed of three rotations
RyModel: each layer is composed of just one RY rotation
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from ._gates_and_grads import RX, RY, RZ, grad_RX, grad_RY, grad_RZ


class ParameterError(Exception):
    """Exception raised when the number of parameters given does
    not correspond with the circuit ansatz."""

    def __init__(self, params_per_layer):
        self.message = f"The number of parameters must be {params_per_layer} per layer."
        super().__init__(self.message)


class Model(ABC):
    """
    Quantum circuit that encodes our function. The circuit consists of
    a number of layers,

    U = Ln * ... * L1
    """

    __slots__ = "encoding", "grad", "params_layer", "__dict__"

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

        self.params_layer = 0  # To be defined in subclasses

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, new_x):
        """Change x doing object.x = new_x"""
        self._x = new_x

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
        w, θ = self.split(params)
        return self.encoding(θ, w)

    @abstractmethod
    def split(self, params: ndarray) -> tuple[ndarray, ndarray]:
        ...

    @abstractmethod
    def _layer(self, θ: ndarray, w: float) -> ndarray:
        """Returns the layer of our circuit."""
        ...

    def _amp_encoding(self, θ: ndarray, w: ndarray) -> ndarray:
        """Returns approximate function encoded in the amplitude of the qubit."""
        U = self._layer(θ[0, :], w[0])[:, :, 0]
        for i in range(1, w.size):
            Ui = self._layer(θ[i, :], w[i])
            U = np.einsum("gmn, gn -> gm", Ui, U)
        return U[:, 0]

    def _prob_encoding(self, θ: ndarray, w: ndarray) -> ndarray:
        """Returns approximate function encoded in the probability of the qubit."""
        fn_amp = self._amp_encoding(θ, w)
        return fn_amp.real**2 + fn_amp.imag**2

    @abstractmethod
    def _grad_layer(self, θ: ndarray, w: float) -> ndarray:
        """Returns the derivative of one layer with respect to its 4 parameters."""
        ...

    def _grad_amp(self, params: ndarray) -> tuple[ndarray, ndarray]:
        """Returns the gradient of the amplitude encoding and the encoded function."""
        w, θ = self.split(params)
        layer_params = 1 + θ.shape[1]
        layers = w.size
        U = np.tensordot(np.ones(self.x.size), np.array([1, 0]), axes=0)  # dim (G,2)
        D = np.zeros((layers, layer_params, self.x.size, 2), dtype=np.complex128)

        for i in range(layers):
            DUi = self._grad_layer(θ[i, :], w[i])  # dim (4,G,2)
            # j is each of the derivatives
            D[i, ...] = np.einsum("jgmn, gn -> jgm", DUi, U)
            # Multiply derivative times next layer
            Ui = self._layer(θ[i, :], w[i])
            U = np.einsum("gmn, gn -> gm", Ui, U)

        grad = np.zeros((layers, layer_params, self.x.size), dtype=np.complex128)
        grad[layers - 1] = D[layers - 1, :, :, 0]
        # In the first iteration we reuse the L-th layer
        B = Ui[:, 0, :]
        for i in range(layers - 2, -1, -1):
            grad[i, ...] = np.einsum("gm, jgm -> jg", B, D[i, ...])
            # Multiply derivative times previous layer
            Ui = self._layer(θ[i, :], w[i])
            B = np.einsum("gn, gnm -> gm", B, Ui)

        grad = np.einsum("ijg -> gij", grad)
        grad = grad.reshape(self.x.size, -1)  # D has shape (x, L*4)
        fn_approx = U[:, 0]

        return grad, fn_approx

    def _grad_prob(self, params: ndarray) -> tuple[ndarray, ndarray]:
        """Returns the gradient of the probability encoding and the encoded function."""
        grad_amp, amp = self._grad_amp(params)
        fn_approx = amp.real**2 + amp.imag**2
        grad_prob = 2 * np.real(np.einsum("g, gi -> gi", amp.conj(), grad_amp))
        return grad_prob, fn_approx


class RotationsModel(Model):
    """
    Each layer of the circuit is made of three rotations dependent
    on 4 parameters:

    L = RX(x * w + θ0) RY(θ1) RZ(θ2)
    """

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
        super().__init__(x, encoding)
        self.params_layer = 4

    def split(self, params: ndarray) -> tuple[ndarray, ndarray]:
        """Split the parameters into"""
        if params.size % self.params_layer != 0:
            raise ParameterError(self.params_layer)
        layers = params.size // self.params_layer
        params = params.reshape(layers, self.params_layer)
        return params[:, 0].reshape(-1), params[:, 1 : self.params_layer]

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
        # move the x axis to first position
        return np.einsum("mn, np, pqg -> gmq", RZ(θ[2]), RY(θ[1]), RX(w * self.x + θ[0]))

    def _grad_layer(self, θ: ndarray, w: float) -> ndarray:
        """Returns the derivative of one layer with respect to its 4 parameters."""
        Dx = np.einsum("mn, np, pqg -> gmq", RZ(θ[2]), RY(θ[1]), grad_RX(w * self.x + θ[0]))
        Dw = np.einsum("gmq, g -> gmq", Dx, self.x)
        Dy = np.einsum("mn, np, pqg -> gmq", RZ(θ[2]), grad_RY(θ[1]), RX(w * self.x + θ[0]))
        Dz = np.einsum("mn, np, pqg -> gmq", grad_RZ(θ[2]), RY(θ[1]), RX(w * self.x + θ[0]))

        return np.array([Dw, Dx, Dy, Dz])  # type: ignore


class RyModel(Model):
    """
    Each layer of the circuit is made of three rotations dependent
    on 2 parameters:

    L = RY(x * w + θ0)
    """

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
        super().__init__(x, encoding)
        self.params_layer = 2

    def split(self, params: ndarray) -> tuple[ndarray, ndarray]:
        """Split the parameters into"""
        if params.size % self.params_layer != 0:
            raise ParameterError(self.params_layer)
        layers = params.size // self.params_layer
        params = params.reshape(layers, self.params_layer)
        return params[:, 0].reshape(-1), params[:, 1 : self.params_layer]

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
        # move the x axis to first position
        return np.einsum("mng -> gmn", RY(w * self.x + θ[0]))

    def _grad_layer(self, θ: ndarray, w: float) -> ndarray:
        """Returns the derivative of one layer with respect to its 2 parameters."""
        Dy = np.einsum("mng -> gmn", grad_RY(w * self.x + θ[0]))
        Dw = np.einsum("gmn, g -> gmn", Dy, self.x)

        return np.array([Dw, Dy])  # type: ignore


class RxRyModel(Model):
    """
    Each layer of the circuit is made of three rotations dependent
    on 3 parameters:

    L = RX(θ0) RY(w * self.x + θ1)
    """

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
        super().__init__(x, encoding)
        self.params_layer = 3

    def split(self, params: ndarray) -> tuple[ndarray, ndarray]:
        """Split the parameters into"""
        if params.size % self.params_layer != 0:
            raise ParameterError(self.params_layer)
        layers = params.size // self.params_layer
        params = params.reshape(layers, self.params_layer)
        return params[:, 0].reshape(-1), params[:, 1 : self.params_layer]

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
        # move the x axis to first position
        return np.einsum("mng, np -> gmp", RY(w * self.x + θ[1]), RX(θ[0]))

    def _grad_layer(self, θ: ndarray, w: float) -> ndarray:
        """Returns the derivative of one layer with respect to its 3 parameters."""

        def _grad_layer_jit(x: ndarray, θ: ndarray, w: float) -> ndarray:
            Dx = np.einsum("mng, np -> gmp", RY(w * x + θ[1]), grad_RX(θ[0]))
            Dy = np.einsum("mng, np -> gmp", grad_RY(w * x + θ[1]), RX(θ[0]))
            Dw = np.einsum("gmn, g -> gmn", Dy, x)
            return np.array([Dw, Dx, Dy])  # type: ignore

        return _grad_layer_jit(self.x, θ, w)
