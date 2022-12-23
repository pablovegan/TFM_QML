"""
Module docstrings
"""
from typing import Callable

import numpy as np
from numpy import cos, sin


class CostFunction:
    """Docstrings"""
    
    def __init__(self, model, x, fn_exact: np.ndarray, metric: str):
        """        
        Parameters
        ----------
        model:
            
        x: ndarray
            The grid in which we approximate our function.
        fn_exact: ndarray
            The value of our function in the grid.
        metric: str
            The metric to measure the difference between the circuit and target state.
        """
        self.model = model
        self.x = x
        self.fn_exact = fn_exact
        if metric == 'mse':
            self.metric = CostFunction.mse_error
        elif metric == 'rmse':
            self.metric = CostFunction.rmse_error
        else:
            ValueError(f"La métrica {metric} introducida no es válida.")
                
    @staticmethod
    def mse_error(fn_exact: np.ndarray, fn_approx: np.ndarray) -> float:
        return np.mean(np.absolute(fn_exact - fn_approx)**2)
    
    @staticmethod
    def rmse_error(fn_exact: np.ndarray, fn_approx: np.ndarray) -> float:
        return np.sqrt(np.mean(np.absolute(fn_exact - fn_approx)**2))

    def __call__(self, θ: np.ndarray, w: np.ndarray):
            """Returns the cost function: MSE or RMSE."""
            fn_approx = self.evaluate_model(self.x, θ, w)


    def der_layer(self, θi: np.ndarray, wi: float) -> tuple:
        """"Returns the derivative of one layer with respect to its 4 parameters."""
        
        ϕ = wi * self.x + θi[0] * np.ones(self.G)
        Rx = np.array([[cos(ϕ/2), -1j * sin(ϕ/2)], [-1j * sin(ϕ/2), cos(ϕ/2)]])
        Ry = np.array([[cos(θi[1]/2), - sin(θi[1]/2)], [sin(θi[1]/2), cos(θi[1]/2)]])
        Rz = np.array([[cos(θi[2]/2) - 1j * sin(θi[2]/2), 0], [0, np.cos(θi[2]/2) + 1j * sin(θi[2]/2)]])

        DRx = 0.5 * np.asarray([[-sin(ϕ/2), -1j * cos(ϕ/2)], [-1j * cos(ϕ/2), -sin(ϕ/2)]])
        DRy = 0.5 * np.array([[-sin(θi[1]/2), -cos(θi[1]/2)], [cos(θi[1]/2), -sin(θi[1]/2)]])
        DRz = 0.5 * np.array([[-1j * cos(θi[2]/2) -sin(θi[2]/2), 0], [0, 1j * cos(θi[2]/2) - sin(θi[2]/2)]])

        Dx = np.einsum('mn,np,pqi->imq', Rz, Ry, DRx)
        Dw = np.einsum('mn,np,pqi,i->imq', Rz, Ry, DRx, self.x)
        Dy = np.einsum('mn,np,pqi->imq', Rz, DRy, Rx)
        Dz = np.einsum('mn,np,pqi->imq', DRz, Ry, Rx)

        return np.array([Dw, Dx, Dy, Dz])

    def der_model(self, θ: np.ndarray, w: np.ndarray):
        """"Create recursively the derivatives with respect to each parameter of the entire net. """
        L = w.size
        A = np.tensordot(np.ones(self.G), np.identity(2), axes=0)  # dim (G,2,2)
        D = np.zeros((L,4,self.G,2,2), dtype=np.complex128)
        for i in range(L):  
            DUi = self.der_layer(θ[:,i], w[i]) # dim (4,G,2,2)
            D[i,...] = np.einsum('jimn,inp->jimp', DUi, A)  # j is each of the derivatives
            # Multiply derivative times next layer
            Ui = self.layer(θ[:,i], w[i])
            A = np.einsum('imn,inp->imp', Ui, A)
        # In the first iteration we reuse the L-th layer
        B = Ui  
        for i in range(L-2,-1,-1):
            D[i,...] = np.einsum('imn,jinp->jimp', B, D[i,...]) 
            # Multiply derivative times previous layer
            Ui = self.layer(self.x, θ[:,i], w[i])
            B = np.einsum('imn,inp->imp', B, Ui)
        return D, A # D is shape (L,4,G,2,2). We also return the model

    def der_cost(self, θ: np.ndarray, w: np.ndarray, return_cost = False):
        """"
        Returns the gradient of the cost function with respect to each parameter. The derivative
        depends on the encoding (probability/amplitude) and the cost function (MSE/RMSE).
        
        Parameters
        ----------
        return_cost : bool
            If True, return the gradient as well as the cost function.

        """
        ders, A = self.der_model(θ, w)
        U = A[:,0,0]

        if self.probability:
            E = (U*np.conj(U)).real - self.f
            if self.cost_fun == 'MSE':
                der_C = 4/self.G * np.array([[np.dot(E.real, np.real(np.conj(U)*ders[i,j,:,0,0])) for i in range(self.L)] for j in range(4)])
            else:
                der_C = 2/(np.sqrt(self.G)*np.sqrt(np.sum(np.abs(E)**2)+1e-9)) * np.array([[np.dot(E, np.real(np.conj(U)*ders[i,j,:,0,0])) for i in range(self.L)] for j in range(4)])
        else:
            E = U - self.f   # error in approximation
            if self.cost_fun == 'MSE':
                der_C = 2/self.G * np.array([[np.real(np.dot(np.conj(E), ders[i,j,:,0,0])) for i in range(self.L)] for j in range(4)])
            else:
                der_C = 1/(np.sqrt(self.G)*np.sqrt(np.sum(np.abs(E)**2)+1e-9)) * np.array([[np.real(np.dot(np.conj(E), ders[i,j,:,0,0])) for i in range(self.L)] for j in range(4)])
        # devolvemos un array con la misma estructura que ϕ = [w, θ_0, θ_1, θ_2]
        if return_cost:
            if self.probability:
                return der_C.flatten(), np.mean((np.abs(U*np.conjugate(U)) - self.f)**2)
            else:
                return der_C.flatten(), np.mean(np.abs(U - self.f)**2)
        else:
            return der_C.flatten()
    