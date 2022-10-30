# coding=UTF-8

import numpy as np
from scipy.integrate import quad
from typing import Optional
import pickle

import qml_model
from test_functions import Test_Functions
from qml_optimizer import QML_optimizer

class QML_results(object):
    def __init__(self,
                function_type: str = 'gaussian',
                f_params: dict = {'mean': 0.0, 'std': 2, 'coef': 1},
                grid_size = 31,
                interval: tuple = (-1,1)
                ):
        self.function = getattr(Test_Functions, function_type)
        self.f_params = f_params
        self.grid_size = grid_size
        self.interval = interval

    def error_perceptron(self, φ: np.ndarray, probability: bool = True):
        """"
        Error in the approximation of the function by the qubit perceptron.
        Returns the error measured in different norms.
        
        """
        layers = φ.size // 4
        w, θ = φ[0:layers], φ[layers:].reshape(3, layers)

        fun = lambda x: self.function(x, **self.f_params)

        # Norma infinito
        y = np.linspace(self.interval[0], self.interval[1], 10000)
        error_inf = np.max(np.abs(fun(y) - qml_model.evaluate_model(y, θ, w).real))

        # Norma L2
        diff_l2 = lambda x: (fun(x) - qml_model.evaluate_model(x, θ, w).real)**2
        error_l2 = np.sqrt(quad(diff_l2, self.interval[0], self.interval[1], limit=100))[0]
        f2_theo = lambda x: fun(x)**2

        # Norma L1
        diff_abs = lambda x: np.abs(fun(x) - qml_model.evaluate_model(x, θ, w, probability).real)
        error_l1 = quad(diff_abs, self.interval[0], self.interval[1], limit=100)[0]

        # Fidelity (no hace falta conjugar f porque el modelo es real)
        prod_re = lambda x: np.real(fun(x)*qml_model.evaluate_model(x, θ, w, probability))
        prod_im = lambda x: np.imag(fun(x)*qml_model.evaluate_model(x, θ, w, probability))
        f2_approx = lambda x: np.abs(qml_model.evaluate_model(x, θ, w, probability))**2
        int_prod_squared = quad(prod_re, self.interval[0], self.interval[1], limit=100)[0]**2+quad(prod_im, self.interval[0], self.interval[1], limit=100)[0]**2
        error_infid = 1 - int_prod_squared/(quad(f2_approx, self.interval[0], self.interval[1], limit=100)[0] * quad(f2_theo, self.interval[0], self.interval[1], limit=100))[0]
        
        return error_l2[0], error_l1[0], error_inf, error_infid


    def graficas_errores(min_layers, max_layers,
                        opt_method: str = 'L-BFGS-B',
                        seed: int = 4,
                        φ_init: Optional[np.ndarray] = None,
                        show_diff = False,
                        print_cost: bool = False,
                        cost_fun: str = 'sqrt',
                        incremental_opt: bool = True,
                        initial_coef: float = 0.3,
                        new_layer_position: str = 'random',
                        new_layer_coef: float = 0.2,
                        method_params: dict = {'n_iter': 800, 'alpha': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8}):
        
        l2_list, l1_list, inf_list, infid_list = [], [], [], []
        layer_list = list(range(min_layers, max_layers+1))
        
        np.random.seed(seed)
        if φ_init is None:
            φ = initial_coef * np.random.randn(min_layers + 3*min_layers)
        else: 
            φ = φ_init

        max_diff = np.zeros(max_layers-min_layers+1)
        cost_error = np.zeros(max_layers-min_layers+1)

        for i, layer in enumerate(layer_list):
            φ, result = qml_optimizer.train_perceptron(x, f, layers = layer, probability = probability, opt_method = opt_method , seed = seed,
                                    φ_init = φ, show_plot = show_plot, method_params = method_params, print_cost = print_cost,
                                    plot_title = function + ' optimized with ' + opt_method, cost_fun = cost_fun)
            # print('Los parámetros óptimos en la capa {layer} son {φ}.\n'.format(layer = layer, φ=φ))
            error_l2, error_l1, error_inf, error_infid = error_perceptron(φ, function, f_params, interval, int_method, probability)
            
            w, θ = qml_optimizer.split(φ)
            cost_error[i] = qml_model.cost(x, f, θ,w, probability)
            # Guardamos la diferencia entre el φ optimizado de la anterior capa y de esta
            if show_diff and (layer > min_layers):
                if new_layer_position == 'final':
                    diff_φ = φ.reshape(4,layer).T.flatten()[0:4*(layer-1)] - φ_old
                elif new_layer_position == 'initial':
                    diff_φ = φ.reshape(4,layer).T.flatten()[4:4*layer] - φ_old
                max_diff[i] = np.abs(diff_φ).max()
                print('La diferencia entre los parámetros optimizados es {diff} y su máximo es {max_diff}.\n'.format(diff = diff_φ, max_diff = max_diff[i]))
            φ_old = φ

            l2_list.append(error_l2)
            l1_list.append(error_l1)
            inf_list.append(error_inf)
            infid_list.append(error_infid)

            if layer == max_layers:
                break

            if incremental_opt is True:
                # Inicializamos una nueva capa en la posición indicada
                if new_layer_position == 'random':
                    i = np.random.randint(0, high=layer+1, dtype=int)
                elif new_layer_position == 'final':
                    i = layer
                elif new_layer_position == 'initial':
                    i = 0
                elif new_layer_position == 'middle':
                    i = min_layers + (layer-min_layers)//2 
                else: raise ValueError('El valor de new_layer_position = {a} no es válido.'.format(a = new_layer_position))
                # Añadimos la nueva capa con valores cercanos a 0
                new_layer_val = new_layer_coef * np.random.randn(4)
                #new_layer_val = 0.3/(i+1) * np.random.randn(4)
                φ = np.insert(φ, i, new_layer_val[0])  # phi [w1, ...wn, theta1, theta2, theta3]
                φ = np.insert(φ, i+1+layer, new_layer_val[1])
                φ = np.insert(φ, i+2+2*layer, new_layer_val[2])
                φ = np.insert(φ, i+3+3*layer, new_layer_val[3])

            else:
                φ = initial_coef * np.random.randn(layer+1 + 3*layer+3)
            # print('Los parámetros con capa añadida son {φ}.\n'.format(φ=φ))

        return layer_list, l2_list, l1_list, inf_list, infid_list, cost_error

    def mean_seed_errores(min_layers, max_layers,
                        opt_method: str = 'L-BFGS-B',
                        φ_init: Optional[np.ndarray] = None,
                        show_plot: bool = False,
                        show_final_plot: bool = True,
                        show_error_plot: bool = True,
                        show_box_plot = True,
                        show_diff = False,
                        print_cost: bool = False,
                        cost_fun: str = 'sqrt',
                        incremental_opt: bool = True,
                        print_params: bool = True,
                        cc: float = 0.3,
                        new_layer_position: str = 'random',
                        new_layer_coef: float = 10e-4,
                        plot_cost_error: bool = False,
                        num_seed = 15,
                        filename = '',
                        method_params: dict = {'n_iter': 800, 'alpha': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8}):          

        num_layer = max_layers - min_layers + 1
        l2, l1, inf, fid, cost = np.zeros(num_layer), np.zeros(num_layer), np.zeros(num_layer), np.zeros(num_layer), np.zeros(num_layer)

        cost_array = np.zeros((num_seed,num_layer))
        l1_array = np.zeros((num_seed,num_layer))
        l2_array = np.zeros((num_seed,num_layer))
        fid_array = np.zeros((num_seed,num_layer))
        inf_array = np.zeros((num_seed,num_layer))

        for i, seed in enumerate(np.random.choice(range(0,100), num_seed, replace=False)):
            layer_list, l2_list, l1_list, inf_list, fid_list, cost_list = graficas_errores(min_layers = min_layers, max_layers = max_layers, x = x, f = f, grid_size = grid_size, function = function, 
                f_params = f_params,interval = interval,int_method = int_method,opt_method = opt_method,φ_init = φ_init,
                show_plot = show_plot, show_final_plot = show_final_plot,show_error_plot = show_error_plot,show_diff = show_diff,
                print_cost = print_cost,cost_fun = cost_fun,incremental_opt = incremental_opt,print_params = print_params, cc = cc,
                new_layer_position = new_layer_position,new_layer_coef = new_layer_coef,plot_cost_error = plot_cost_error,
                method_params=method_params, seed = seed)

            # Seeds en el eje 0 y capas en el eje 1. Queremos las seeds en cada box plot.'''
            cost_array[i,:]= np.array(cost_list)
            l1_array[i,:]= np.array(l1_list)
            l2_array[i,:]= np.array(l2_list)
            fid_array[i,:]= np.array(fid_list)
            inf_array[i,:]= np.array(inf_list)

        with open(filename+'.pkl', 'wb') as file:
            pickle.dump((layer_list, l2_array, l1_array, inf_array, fid_array, cost_array), file)

        return