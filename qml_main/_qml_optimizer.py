# coding=UTF-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Optional
import qml_model


@staticmethod
def split(φ):
    layers = φ.size // 4
    return φ[0:layers], φ[layers:].reshape(3, layers)

def blackbox_minimizer(x, f, φ_init, probability: bool,
             opt_method: str = 'L-BFGS-B', print_cost: bool = False, cost_fun = 'sqrt'):
    
    if cost_fun == 'sqrt':
        cost_function = globals()["coste_sqrt"]
        der_cost_function = globals()["der_coste_sqrt"]
    elif cost_fun == 'normal':
        cost_function = globals()["coste"]
        der_cost_function = globals()["der_coste"]

    def intermediate_cost(φ):
        w, θ = split(φ)
        c = cost_function(x, f, θ, w, probability)
        if print_cost:  print('Valor f. coste: ', c)
        return c

    def intermedio_der_cost(φ):
        w, θ = split(φ)
        der_c = der_cost_function(x, f, θ, w, probability)
        if print_cost:  print('Valor der. coste: ', der_c)
        return der_c

    return minimize(intermediate_cost, φ_init, method = opt_method, jac = intermedio_der_cost, tol = 1e-12, options={'maxiter': 10000})

def adam(x, f, φ, probability: bool, print_cost: bool = True, plot_cost = False, cost_fun = 'sqrt', n_iter = 800,
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
	if cost_fun == 'sqrt':
		der_cost_function = globals()["der_coste_sqrt"]
	elif cost_fun == 'normal':
		der_cost_function = globals()["der_coste"]

	num_params = φ.size
	# initialize first and second moments
	m = np.zeros(num_params)
	v = np.zeros(num_params)
	# Model parameters
	min_cost = 10
	cost = np.zeros(n_iter)
	for t in range(n_iter):
		w, θ = split(φ)
		# g, cost = der_coste(x, f, θ, w, probability, return_cost = True)
		g, cost[t] = der_cost_function(x, f, θ, w, probability, return_cost = True)
		if cost[t] < min_cost: 
			min_cost = cost[t]
			min_t = t
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
def train_perceptron(x: np.ndarray, f: np.ndarray,
                     layers: int = 4,
                     opt_method: str = 'L-BFGS-B',
                     method_params: dict = {'n_iter': 800, 'alpha': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8},
                     seed: float = 2.0,
                     φ_init: Optional[np.ndarray] = None,
                     print_cost: bool = False,
                     show_plot = True,
                     cc = 0.3,
                     probability: Optional[np.ndarray] = None,
                     plot_cost: bool = False,
                     cost_fun: str = 'sqrt',
                     plot_title: str = '' ):

    if φ_init is None:
        np.random.seed(seed) 
        φ_init = cc*np.random.randn(layers + 3*layers)

    if probability is None:
        probability = (f >= 0).all()
    
    if opt_method == 'ADAM':
        φ = adam(x,f, φ_init, probability = probability, print_cost = print_cost, plot_cost = plot_cost,
                    cost_fun = cost_fun, **method_params)
        result = 0  # resultado a 0 por defecto en este método
    else:
        result = blackbox_minimizer(x, f, φ_init, opt_method = opt_method, probability = probability,
        print_cost = print_cost, cost_fun = cost_fun)
        φ = result.x

    if show_plot:
        ω, θ = split(φ) 
        f_approx = qml_model.evaluate_model(x, θ, ω, probability)
        plt.close('all')
        plt.plot(x, f)
        plt.plot(x, f_approx.real)
        plt.title(plot_title)
        plt.show()
    
    return φ, result