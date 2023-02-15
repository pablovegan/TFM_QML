"""
Module docstrings

Classes
-------
Circuit: abstract class, it is the basis of our specific circuit classes.
CircuitRxRyRz: each layer is composed of three rotations
CircuitRxRy: each layer is composed of RX and RY rotation
CircuitRy: each layer is composed of just one RY rotation
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


class Circuit(ABC):
    """Quantum circuit that encodes the function. The circuit
    consists of a number of layers,

    U = Ln * ... * L1

    Attributes
    ----------
    encoding : Callable
        Return the encoding of the function in the circuit. For example
        amplitude or probability of the |0> qubit.
    grad_encoding : Callable
        Returns the gradient of the chosen encoding.
    params_layer : int
        Number of parameters per layer.
    """

    __slots__ = "encoding", "grad", "params_layer", "__dict__"

    def __init__(self, x: ndarray, encoding_type: str, params_layer: int):
        """
        Parameters
        ----------
        x: ndarray
            Values where to evaluate the function encoded in the circuit.
        encoding_type : str
            Choose between amplitude or probability encoding.
            Must be either 'amp' or 'prob'.
        params_layer : int
            Number of parameters per layer.
        """
        self.x = x

        if encoding_type == "prob":
            self.encoding = self.prob_encoding
            self.grad_encoding = self.grad_prob
        elif encoding_type == "amp":
            self.encoding = self.amp_encoding
            self.grad_encoding = self.grad_amp
        else:
            raise ValueError("Invalid encoding '{encoding}'. Choose between 'prob' or 'amp'.")

        self.params_layer = params_layer  # To be defined in subclasses

    @property
    def x(self) -> ndarray:
        """Values where to evaluate the function encoded in the circuit.

        Returns
        -------
        ndarray
            The value of x.
        """
        return self._x

    @x.setter
    def x(self, new_x: ndarray) -> None:
        """
        Update the array where to evaluate the function.

        Parameters
        ----------
        new_x : ndarray
            New value for the independent variable x.
        """
        self._x = new_x

    @abstractmethod
    def layer(self, params: ndarray) -> ndarray:
        """Returns the layer of our circuit

        Parameters
        ----------
        params : ndarray
            Parameters of the quantum gates in the layer.

        Returns
        -------
        ndarray
            Unitary matrix of the layer with size (x,2,2)
        """
        ...

    def amp_encoding(self, params: ndarray) -> ndarray:
        """Returns approximate function encoded in the amplitude of the qubit.

        Parameters
        ----------
        params : ndarray
            Parameters of the quantum gates in the layer.

        Returns
        -------
        ndarray
            Values of the amplitudes of the |0> qubit for each value of x.
        """
        layers = params.size // self.params_layer
        params = params.reshape(layers, self.params_layer)
        U = self.layer(params[0, :])[:, :, 0]
        for i in range(1, params.shape[0]):
            Ui = self.layer(params[i, :])
            U = np.einsum("gmn, gn -> gm", Ui, U)
        return U[:, 0]

    def prob_encoding(self, params: ndarray) -> ndarray:
        """Returns approximate function encoded in the probability of the qubit.
        s
                Parameters
                ----------
                params : ndarray
                    Parameters of the quantum gates in the layer.

                Returns
                -------
                ndarray
                    Values of the probabilities of the |0> qubit for each value of x.
        """
        fn_amp = self.amp_encoding(params)
        return fn_amp.real**2 + fn_amp.imag**2

    @abstractmethod
    def grad_layer(self, params: ndarray) -> ndarray:
        """Returns the derivative of one layer with respect to its parameters.

        Parameters
        ----------
        params : ndarray
            Parameters of the quantum gates in the layer.

        Returns
        -------
        ndarray
            Values of the probabilities of the |0> qubit for each value of x.
        """
        ...

    def grad_amp(self, params: ndarray) -> tuple[ndarray, ndarray]:
        """Returns the gradient of the amplitude encoding and the encoded function.

        Parameters
        ----------
        params : ndarray
            Parameters of the quantum gates in the layer.

        Returns
        -------
        tuple[ndarray, ndarray]
            Gradients of the amplitude with respect to all parameters and the amplitudes for each x.
        """
        layers = params.size // self.params_layer
        params = params.reshape(layers, self.params_layer)
        U = np.tensordot(np.ones(self.x.size), np.array([1, 0]), axes=0)  # dim (G,2)
        D = np.zeros((layers, self.params_layer, self.x.size, 2), dtype=np.complex128)

        for i in range(layers):
            DUi = self.grad_layer(params[i, :])  # dim (4,G,2)
            # j is each of the derivatives
            D[i, ...] = np.einsum("jgmn, gn -> jgm", DUi, U)
            # Multiply derivative times next layer
            Ui = self.layer(params[i, :])
            U = np.einsum("gmn, gn -> gm", Ui, U)

        grad = np.zeros((layers, self.params_layer, self.x.size), dtype=np.complex128)
        grad[layers - 1] = D[layers - 1, :, :, 0]
        # In the first iteration we reuse the L-th layer
        B = Ui[:, 0, :]
        for i in range(layers - 2, -1, -1):
            grad[i, ...] = np.einsum("gm, jgm -> jg", B, D[i, ...])
            # Multiply derivative times previous layer
            Ui = self.layer(params[i, :])
            B = np.einsum("gn, gnm -> gm", B, Ui)

        grad = np.einsum("ijg -> gij", grad)
        grad = grad.reshape(self.x.size, -1)  # D has shape (x, L*4)
        fn_approx = U[:, 0]

        return grad, fn_approx

    def grad_prob(self, params: ndarray) -> tuple[ndarray, ndarray]:
        """Returns the gradient of the probability encoding and the probability encoding.

        Parameters
        ----------
        params : ndarray
            Parameters of the quantum gates in the layer.

        Returns
        -------
        tuple[ndarray, ndarray]
            Gradients of the probability with respect to all parameters and the probability for each x.
        """
        grad_amp, amp = self.grad_amp(params)
        fn_approx = amp.real**2 + amp.imag**2
        grad_prob = 2 * np.real(np.einsum("g, gi -> gi", amp.conj(), grad_amp))
        return grad_prob, fn_approx


class CircuitRxRyRz(Circuit):
    """
    Each layer of the circuit is made of three rotations dependent
    on 4 parameters:

    L = RX(x * w + θx) RY(θy) RZ(θz)
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
        self.params_layer = 4
        super().__init__(x, encoding, self.params_layer)

    def layer(self, params: ndarray) -> ndarray:
        """
        Returns the layer of the circuit:
        L = RX(x * w + θ0) RY(θ1) RZ(θ2)

        Parameters
        ----------
        params : ndarray
            Parameters of the quantum gates in the layer.

        Returns
        -------
        ndarray
            Unitary matrix of the layer with size (x,2,2)

        Raises
        ------
        ParameterError
            The number of parameters given does not correspond with
            the circuit ansatz.
        """
        if params.size != self.params_layer:
            raise ParameterError(self.params_layer)
        w = params[0]
        θx = params[1]
        θy = params[2]
        θz = params[3]
        # move the x axis to first position
        return np.einsum("mn, np, pqg -> gmq", RZ(θz), RY(θy), RX(w * self.x + θx))

    def grad_layer(self, params: ndarray) -> ndarray:
        """Returns the derivative of one layer with respect to its 4 parameters.

        Parameters
        ----------
        params : ndarray
            Parameters of the quantum gates in the layer.

        Returns
        -------
        ndarray
            Gradient of the layer with respect to each parameter.
        """
        w = params[0]
        θx = params[1]
        θy = params[2]
        θz = params[3]

        Dx = np.einsum("mn, np, pqg -> gmq", RZ(θz), RY(θy), grad_RX(w * self.x + θx))
        Dw = np.einsum("gmq, g -> gmq", Dx, self.x)
        Dy = np.einsum("mn, np, pqg -> gmq", RZ(θz), grad_RY(θy), RX(w * self.x + θx))
        Dz = np.einsum("mn, np, pqg -> gmq", grad_RZ(θz), RY(θy), RX(w * self.x + θx))

        return np.array([Dw, Dx, Dy, Dz])  # type: ignore


class CircuitRxRy(Circuit):
    """
    Each layer of the circuit is made of three rotations dependent
    on 3 parameters:

    L = RX(θx) RY(w * x + θy)
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
        self.params_layer = 3
        super().__init__(x, encoding, self.params_layer)

    def layer(self, params: ndarray) -> ndarray:
        """
        Each layer is the product of two rotations.
        L = RX(θx) RY(w * x + θy)

        Parmeters
        ---------
        params : ndarray
            Parameters of the gates in the layer.

        Returns
        -------
        ndarray
            Unitary matrix of the layer with size (x,2,2)

        Raises
        ------
        ParameterError
            The number of parameters given does not correspond with
            the circuit ansatz.
        """
        if params.size != self.params_layer:
            raise ParameterError(self.params_layer)
        w = params[0]
        θx = params[1]
        θy = params[2]
        # move the x axis to first position
        return np.einsum("mng, np -> gmp", RY(w * self.x + θy), RX(θx))

    def grad_layer(self, params: ndarray) -> ndarray:
        """Returns the derivative of one layer with respect to its 3 parameters.

        Parameters
        ----------
        params : ndarray
            Parameters of the quantum gates in the layer.

        Returns
        -------
        ndarray
            Gradient of the layer with respect to each parameter.
        """
        w = params[0]
        θx = params[1]
        θy = params[2]
        Dx = np.einsum("mng, np -> gmp", RY(w * self.x + θy), grad_RX(θx))
        Dy = np.einsum("mng, np -> gmp", grad_RY(w * self.x + θy), RX(θx))
        Dw = np.einsum("gmn, g -> gmn", Dy, self.x)
        return np.array([Dw, Dx, Dy])  # type: ignore


class CircuitRy(Circuit):
    """
    Each layer of the circuit is made of three rotations dependent
    on 2 parameters:

    L = RY(w * x + θy)
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
        self.params_layer = 2
        super().__init__(x, encoding, self.params_layer)

    def layer(self, params: ndarray) -> ndarray:
        """
        Each layer is one RY rotation:
        L = RY(w * x + θy)

        Parmeters
        ---------
        params : ndarray
            Parameters of the gates in the layer.

        Returns
        -------
        ndarray
            Unitary matrix of the layer with size (x,2,2)

        Raises
        ------
        ParameterError
            The number of parameters given does not correspond with
            the circuit ansatz.
        """
        if params.size != self.params_layer:
            raise ParameterError(self.params_layer)
        w = params[0]
        θy = params[1]
        # move the x axis to first position
        return np.einsum("mng -> gmn", RY(w * self.x + θy))

    def grad_layer(self, params: ndarray) -> ndarray:
        """Returns the derivative of one layer with respect to its 2 parameters.

        Parameters
        ----------
        params : ndarray
            Parameters of the quantum gates in the layer.

        Returns
        -------
        ndarray
            Gradient of the layer with respect to each parameter.
        """
        w = params[0]
        θy = params[1]
        Dy = np.einsum("mng -> gmn", grad_RY(w * self.x + θy))
        Dw = np.einsum("gmn, g -> gmn", Dy, self.x)

        return np.array([Dw, Dy])  # type: ignore
