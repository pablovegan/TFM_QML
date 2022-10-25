# coding=UTF-8

from qml_model import QML_model

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Optional
from qml_model import QML_model

class QML_optimizer(QML_model):
    def __init__(self, f, x, θ, w, cost_fun, probability = None, 
                opt_method: str = 'L-BFGS-B', return_cost = False):
        super().__init__(f, x, θ, w, cost_fun, probability)
        self.opt_method = opt_method
        self.return_cost = return_cost

    @staticmethod
    def split(φ):
        layers = φ.size // 4
        return φ[0:layers], φ[layers:].reshape(3, layers)

    def blackbox_minimizer(self, φ_init, print_cost: bool = False):

        def intermediate_cost(φ):
            w, θ = QML_optimizer.split(φ)
            c = super().cost(θ, w)
            if print_cost:  print('Valor f. coste: ', c)
            return c

        def intermedio_der_cost(φ):
            w, θ = QML_optimizer.split(φ)
            der_c = super().der_cost(θ, w, return_cost = self.return_cost)
            if print_cost:  print('Valor der. coste: ', der_c)
            return der_c

        return minimize(intermediate_cost, φ_init, method = self.opt_method, jac = intermedio_der_cost, tol = 1e-12, options={'maxiter': 10000})

    def adam(self, φ, print_cost: bool = True, plot_cost = False, n_iter = 800,
            alpha = 0.01, beta1 = 0.9, beta2 = 0.999, eps=1e-8):
        '''
        Parameters
        ----------
        n_iter : int
            Number of iterations of the optimization algorithm
        alpha : float
            steps size
        beta1 : float
            factor for average gradient
        beta2 : float
            factor for average squared gradient

        '''
        num_params = φ.size
        # initialize first and second moments
        m = np.zeros(num_params)
        v = np.zeros(num_params)
        # Model parameters
        min_cost = 100
        cost = np.zeros(n_iter)
        for t in range(n_iter):
            w, θ = QML_optimizer.split(φ)
            g, cost[t] = super().der_cost(θ, w, return_cost = True)
            if cost[t] < min_cost: 
                min_cost = cost[t]
                # min_t = t
                min_φ = φ
            if print_cost:
                print('φ = {φ}  ,  cost = {cost}'.format(φ = φ, cost = cost[t]))
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * g**2
            mhat = m / (1.0 - beta1**(t+1))
            vhat = v / (1.0 - beta2**(t+1))
            φ = φ - alpha * mhat / (np.sqrt(vhat) + eps)
        # print('El coste mínimo alcanzado es {min_cost} en la iteración {min_t}.'.format(min_cost=min_cost, min_t=min_t))
        # Devolvemos el φ que minimiza la función de coste
        if plot_cost:
            plt.plot(range(n_iter), cost)
            plt.yscale('log')
            plt.show()
        return min_φ

    def train_perceptron(self,
                        layers: int = 4,
                        opt_method: str = 'L-BFGS-B',
                        method_params: dict = {'n_iter': 800, 'alpha': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8},
                        seed: float = 2.0,
                        φ_init: Optional[np.ndarray] = None,
                        print_cost: bool = False,
                        show_plot = True,
                        cc = 0.3,
                        plot_cost: bool = False,
                        plot_title: str = '' ):

        if φ_init is None:
            np.random.seed(seed) 
            φ_init = cc*np.random.randn(layers + 3*layers)
        
        if opt_method == 'ADAM':
            φ = self.adam(φ_init, print_cost = print_cost,
                         plot_cost = plot_cost, **method_params)
            result = 0  # resultado a 0 por defecto en este método
        else:
            result = self.blackbox_minimizer(φ_init, opt_method = opt_method, probability = probability,
            print_cost = print_cost)
            φ = result.x

        if show_plot:
            w, θ = QML_optimizer.split(φ) 
            f_approx = super().evaluate_model(θ, w)
            plt.close('all')
            plt.plot(super().x, super().f)
            plt.plot(super().x, f_approx.real)
            plt.title(plot_title)
            plt.show()
        
        return φ, result