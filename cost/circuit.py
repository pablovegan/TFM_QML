"""
Module docstrings
"""

import numpy as np
from numpy import cos, sin


class Model:
    
    def __init__(self, encoding: str):
        """
        Parameters
        ----------
        layers: int
            The number of layers in our circuit.
        encoding: str
            Choose between amplitude or probability encoding.
        """
        if encoding == 'prob':
            self.prob = True
        elif encoding == 'amp':
            self.prob = False
        else:
            raise ValueError("Invalid encoding '{encoding}'. Choose between 'prob' or 'amp'.")

    @staticmethod
    def _layer(x: np.ndarray, θ: np.ndarray, w: float) -> tuple:
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
        ϕ = w * x + θ[0]
        Rx = np.array([[cos(ϕ/2), -1j * sin(ϕ/2)], [-1j * sin(ϕ/2), cos(ϕ/2)]])
        Ry = np.array([[cos(θ[1]/2), -sin(θ[1]/2)], [sin(θ[1]/2), cos(θ[1]/2)]])
        Rz = np.array([[cos(θ[2]/2) - 1j * sin(θ[2]/2), 0], [0, cos(θ[2]/2) + 1j * sin(θ[2]/2)]])

        return np.einsum('mn, np, pqi -> imq', Rz, Ry, Rx) # move the x axis to first position

    def _model(self, x: np.ndarray, θ: np.ndarray, w: np.ndarray):
        """
        Returns our variational ansatz, the product of the L layers.
        Since we are interested in the amplitude/probability of the |0> qubit
        we select the (0,0) element of the unitary matrix U (for every x).
        """
        U = Model._layer(x, θ[:,0], w[0])[:,:,0]
        for i in range(1, w.size):
            Ui = Model._layer(x, θ[:,i], w[i])
            U = np.einsum('imn, in -> im', Ui, U)
        return U[:,0]

    def evaluate(self, x: np.ndarray, θ: np.ndarray, w: np.ndarray):
        """Calculate the amplitude or probability of the |0>, depending on the encoding."""
        assert θ.shape[1] == w.size, (f"Length of w = {w.size}, but must equal",
            f"size of axis 0 in θ with size {θ.shape[1]}.")
        U_00 = self._model(x, θ, w)
        return  U_00.real**2 + U_00.imag**2 if self.prob else U_00

    @staticmethod
    def _der_layer(x: np.ndarray, θ: np.ndarray, w: float) -> tuple:
        """"Returns the derivative of one layer with respect to its 4 parameters."""
        ϕ = w * x + θ[1]

        Rx = np.array([[cos(ϕ/2), -1j * sin(ϕ/2)], [-1j * sin(ϕ/2), cos(ϕ/2)]])
        Ry = np.array([[cos(θ[1]/2), -sin(θ[1]/2)], [sin(θ[1]/2), cos(θ[1]/2)]])
        Rz = np.array([[cos(θ[2]/2) - 1j * sin(θ[2]/2), 0], [0, cos(θ[2]/2)+1j * sin(θ[2]/2)]])

        DRx = 0.5 * np.asarray([[-sin(ϕ/2), -1j * cos(ϕ/2)], [-1j * cos(ϕ/2), -sin(ϕ/2)]])
        DRy = 0.5 * np.array([[-sin(θ[1]/2), - cos(θ[1]/2)], [cos(θ[1]/2), -sin(θ[1]/2)]])
        DRz = 0.5 * np.array([[-1j * cos(θ[2]/2) - sin(θ[2]/2), 0], [0, 1j * cos(θ[2]/2) - sin(θ[2]/2)]])

        Dx = np.einsum('mn, np, pqi -> imq', Rz, Ry, DRx)
        Dw = np.einsum('mn, np, pqi, i -> imq', Rz, Ry, DRx, x)
        Dy = np.einsum('mn, np, pqi -> imq', Rz, DRy, Rx)
        Dz = np.einsum('mn, np, pqi -> imq', DRz, Ry, Rx)

        return np.array([Dw, Dx, Dy, Dz])

    def grad(self, x: np.ndarray, θ: np.ndarray, w: np.ndarray):
        """"Create recursively the derivatives with respect to each parameter of the entire net. """
        assert θ.shape[1] == w.size, (f"Length of w = {w.size}, but must equal",
            f"size of axis 0 in θ with size {θ.shape[1]}.")
        
        layers = w.size
        A = np.tensordot(np.ones(x.size), np.identity(2), axes=0)  # dim (G,2,2)
        D = np.zeros((layers, 4, x.size, 2, 2), dtype = np.complex128)
        
        for i in range(layers):  
            DUi = Model._der_layer(x, θ[:,i], w[i]) # dim (4,G,2,2)
            D[i,...] = np.einsum('jimn, inp -> jimp', DUi, A)  # j is each of the derivatives
            # Multiply derivative times next layer
            Ui = Model._layer(x, θ[:,i], w[i])
            A = np.einsum('imn, inp -> imp', Ui, A)
        # In the first iteration we reuse the L-th layer
        B = Ui  
        for i in range(layers - 2, -1, -1):
            D[i,...] = np.einsum('imn, jinp -> jimp', B, D[i,...]) 
            # Multiply derivative times previous layer
            Ui = Model._layer(x, θ[:,i], w[i])
            B = np.einsum('imn, inp -> imp', B, Ui)
        return D # D is shape (L,4,G,2,2). We also return the model
        return D, A # D is shape (L,4,G,2,2). We also return the model
