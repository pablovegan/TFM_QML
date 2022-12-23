# coding=UTF-8

import numpy as np
from test_functions import Test_Functions

class QML_model(object):
    """
    Instance Attributes
    -------------------
    x : (G) array
        Discretized grid used to approximate our function.
    θ : (L,3) array
        Bias parameters of each layer and rotation.
    w : (L) array
        Weight of the X rotation of each layer.
    probability : bool
        True if the function to approximate is positive

    References
    ----------
    [1] Adrián Pérez Salinas et al, "Data re-uploading for a universal
        quantum classifier Quantum" 4, 226 (2020)
    """
    
    def __init__(self,
                function_type: str = 'gaussian',
                f_params: dict = {'mean': 0.0, 'std': 2, 'coef': 1},
                grid_size = 31,
                cost_fun = 'RMSE',
                probability = None):
        function = getattr(Test_Functions, function_type)
        self.x = x
        self.G = 1 if isinstance(x) is float else x.size
        self.f = f
        self.cost_fun = cost_fun

        if cost_fun != 'MSE' or cost_fun != 'RMSE': 
            raise ValueError('Cost function must be either MSE or RMSE.') 
        else:
            self.cost_fun = cost_fun

        if probability is None:
            self.probability = (f >= 0).all()
        elif isinstance(probability) is not bool:
            raise TypeError('Probability must be boolean.')
        else:
            self.probability = probability

    def layer(self, x: np.ndarray or float, θi: np.ndarray, wi: float) -> tuple:
        """
        Each layer is the product of three rotations.
        
        Parmeters
        ---------
        θi : (3) array
            Bias parameters of each rotation.
        wi : float
            Weight of the X rotation.
        
        Returns
        -------
        A : (G,2,2) array
            Unitary matrix of the layer.

        """
        G = 1 if type(x) is float else x.size
        ϕ1 = wi*x+θi[0]*np.ones(G)
        ϕ2 = θi[1]
        ϕ3 = θi[2]

        Rx = np.asarray([[np.cos(ϕ1/2), -1j*np.sin(ϕ1/2)],[-1j*np.sin(ϕ1/2), np.cos(ϕ1/2)]])
        Ry = np.array([[np.cos(ϕ2/2), -np.sin(ϕ2/2)],[np.sin(ϕ2/2), np.cos(ϕ2/2)]])
        Rz = np.array([[np.cos(ϕ3/2) - 1j*np.sin(ϕ3/2), 0],[0, np.cos(ϕ3/2)+1j*np.sin(ϕ3/2)]])

        Ui = np.einsum('mn,np,pqi->mqi', Rz, Ry, Rx)
        return np.moveaxis(Ui, -1, 0)  # move last axis to the first position keeping the order of the rest axis

    def model(self, x: np.ndarray or float, θ: np.ndarray, w: np.ndarray):
        """
        Returns our variational ansatz, the product of the L layers.
        Since we are interested in the amplitude/probability of the |0> qubit
        we select the (0,0) element of the unitary matrix U (for every x).
        """
        U = self.layer(x, θ[:,0], w[0])[:,:,0]
        for i in range(1, w.size):
            Ui = self.layer(x, θ[:,i], w[i])
            U = np.einsum('imn,in->im', Ui, U)
        return U[:,0]

    def evaluate_model(self, x: np.ndarray or float, θ: np.ndarray, w: np.ndarray):
        """Calculate the amplitude or probability of the |0>, depending on the encoding."""
        U_00 = self.model(x, θ, w)
        return (U_00*np.conjugate(U_00)).real if self.probability else U_00

    def cost(self, θ: np.ndarray, w: np.ndarray):
        """Returns the cost function: MSE or RMSE."""
        f_approx = self.evaluate_model(self.x, θ, w)
        if self.cost_fun == 'MSE':
            return np.mean(np.abs(f_approx - self.f)**2)
        else:
            return np.sqrt(np.mean(np.abs(f_approx - self.f)**2))

    def der_layer(self, θi: np.ndarray, wi: float) -> tuple:
        """"Returns the derivative of one layer with respect to its 4 parameters."""
        ϕ1 = wi*self.x+θi[0]*np.ones(self.G)
        ϕ2 = θi[1]
        ϕ3 = θi[2]

        Rx = np.asarray([[np.cos(ϕ1/2), -1j*np.sin(ϕ1/2)],[-1j*np.sin(ϕ1/2), np.cos(ϕ1/2)]])
        Ry = np.array([[np.cos(ϕ2/2), -np.sin(ϕ2/2)],[np.sin(ϕ2/2), np.cos(ϕ2/2)]])
        Rz = np.array([[np.cos(ϕ3/2) - 1j*np.sin(ϕ3/2), 0],[0, np.cos(ϕ3/2)+1j*np.sin(ϕ3/2)]])

        DRx = 1/2*np.asarray([[-np.sin(ϕ1/2), -1j*np.cos(ϕ1/2)],[-1j*np.cos(ϕ1/2), -np.sin(ϕ1/2)]])
        DRy = 1/2*np.array([[-np.sin(ϕ2/2), -np.cos(ϕ2/2)],[np.cos(ϕ2/2), -np.sin(ϕ2/2)]])
        DRz = 1/2*np.array([[-1j*np.cos(ϕ3/2) - np.sin(ϕ3/2), 0],[0, 1j*np.cos(ϕ3/2)-np.sin(ϕ3/2)]])

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