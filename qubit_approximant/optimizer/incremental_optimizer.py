"""Incremental optimizer"""

from typing import Callable

import numpy as np

from .optimizer import Optimizer, BlackBoxOptimizer, GDOptimizer, AdamOptimizer


class IncrementalOptimizer:
    """This optimizer uses the parameters of an optimized L layer model
    as input for the optimization of a L+1 layer model."""

    blackbox_methods = ["CG", "L-BFGS-B", "COBYLA", "SLSQP"]
    layer_positions = ["initial", "middle", "final", "random"]

    def __init__(self, min_layer, max_layer, new_layer_position: str, method: str, method_kwargs: dict = {}):
        """
        Initialize a black box optimizer.

        Parameters
        ----------
        method: str
            The desired optimization method.
        """
        self.min_layer = min_layer
        self.max_layer = max_layer
        
        if new_layer_position in IncrementalOptimizer.layer_positions:
            self.new_layer_position = new_layer_position
        else:
            raise ValueError(f"new_layer_position = {new_layer_position} is not supported. "
                             "Try 'initial', 'middle', 'final' or 'random'")
        
        if method in IncrementalOptimizer.blackbox_methods:
            self.optimizer: Optimizer = BlackBoxOptimizer(method, method_kwargs)
        elif method == 'GD':
            self.optimizer = GDOptimizer(**method_kwargs)
        elif method == 'Adam':
            self.optimizer = AdamOptimizer(**method_kwargs)
        else:
            raise ValueError(f"Optimization {method} is not supported.")
        
    def __call__(self, cost: Callable, grad_cost: Callable, init_params: np.ndarray) -> list[np.ndarray]:
        
        self.params_per_layer = init_params // self.min_layer
        params = init_params
        params_list = []
        
        for _ in range(self.min_layer, self.max_layer + 1):
            params = self.optimizer(cost, grad_cost, params)
            params_list.append(params)
            
        return params_list
    
    def _new_initial_params(params: np.ndarray) -> np.ndarray:
        """Create new initial parameters from the optimized parameters
        with one layer less."""
        

        if self.new_layer_position == "final":
            i = layer
        elif self.new_layer_position == "middle":
            i = min_layers + (layer - min_layers) // 2
        elif self.new_layer_position == "initial":
            i = 0
        elif self.new_layer_position == "random":
            i = np.random.randint(0, high=layer + 1, dtype=int)
            
        # Añadimos la nueva capa con valores cercanos a 0
        new_layer_val = new_layer_coef * np.random.randn(4)
        # new_layer_val = 0.3/(i+1) * np.random.randn(4)
        φ = np.insert(
            φ, i, new_layer_val[0]
        )  # phi [w1, ...wn, theta1, theta2, theta3]
        φ = np.insert(φ, i + 1 + layer, new_layer_val[1])
        φ = np.insert(φ, i + 2 + 2 * layer, new_layer_val[2])
        φ = np.insert(φ, i + 3 + 3 * layer, new_layer_val[3])
        else:
            φ = cc * np.random.randn(layer + 1 + 3 * layer + 3)
        # print('Los parámetros con capa añadida son {φ}.\n'.format(φ=φ))
