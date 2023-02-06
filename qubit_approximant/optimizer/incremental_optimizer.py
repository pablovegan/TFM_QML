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
        
    def __call__(self, cost: Callable, grad_cost: Callable, init_params: np.ndarray, new_layer_coef: float = 0.3) -> list[np.ndarray]:
        
        self.params_per_layer = init_params // self.min_layer
        params = init_params
        self.params_list = []
        
        for layer in range(self.min_layer, self.max_layer + 1):
            params = self.optimizer(cost, grad_cost, params)
            self.params_list.append(params)
            params = self._new_initial_params(params, layer, new_layer_coef)
        return self.params_list
    
    def _new_initial_params(self, params: np.ndarray, current_layer: int, new_layer_coef: float) -> np.ndarray:
        """Create new initial parameters from the optimized parameters
        with one layer less."""

        if self.new_layer_position == "final":
            layer = current_layer
        elif self.new_layer_position == "middle":
            layer = self.min_layer + (current_layer - self.min_layer) // 2
        elif self.new_layer_position == "initial":
            layer = 0
        elif self.new_layer_position == "random":
            layer = np.random.randint(0, high=current_layer + 1, dtype=int)
            
        new_layer_val = new_layer_coef * np.random.randn(4)
        params = np.insert(params, layer, new_layer_val)  # [w1, ...wn, theta1, theta2, theta3]
        
        return params
    
    def inital_params_diff(self):
        
        mean_diff = []
        std_diff = []
        
        if self.new_layer_position == "final":
            for i in range(self.min_layer, self.max_layer - 1):
                params0 = self.params_list[i]
                params1 = self.params_list[i + 1][0: i * self.params_per_layer]
                params_diff = params1 - params0
                mean_diff.append(np.mean(np.abs(params_diff)))
                std_diff.append(np.std(np.abs(params_diff)))
                
        elif self.new_layer_position == "initial":
            for i in range(self.min_layer, self.max_layer - 1):
                params0 = self.params_list[i]
                params1 = self.params_list[i + 1][self.params_per_layer: i * self.params_per_layer]
                params_diff = params1 - params0
                mean_diff.append(np.mean(np.abs(params_diff)))
                std_diff.append(np.std(np.abs(params_diff)))
        else:
            raise ValueError("Parameter difference only supported for new initial and final layers.")
