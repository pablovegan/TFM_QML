"""Incremental optimizer"""

from typing import Callable

import numpy as np

from .multilayer_optimizer import MultilayerOptimizer


class NonIncrementalOptimizer(MultilayerOptimizer):
    """This optimizer uses the parameters of an optimized L layer model
    as input for the optimization of a L+1 layer model."""

    blackbox_methods = ["CG", "L-BFGS-B", "COBYLA", "SLSQP"]
    layer_positions = ["initial", "middle", "final", "random"]

    def __init__(
        self, min_layer, max_layer, new_layer_position: str, method: str, method_kwargs: dict = {}
    ):
        """
        Initialize a black box optimizer.

        Parameters
        ----------
        method: str
            The desired optimization method.
        """
        super().__init__(min_layer, max_layer, new_layer_position, method, method_kwargs)

    def __call__(
        self,
        cost: Callable,
        grad_cost: Callable,
        init_params: np.ndarray,
        new_layer_coef: float = 0.3,
    ) -> list[np.ndarray]:

        self.params_per_layer = init_params // self.min_layer
        params = init_params
        self.params_list = []

        for layer in range(self.min_layer, self.max_layer + 1):
            params = self.optimizer(cost, grad_cost, params)
            self.params_list.append(params)
            params = new_layer_coef * np.random.randn((layer + 1) * self.params_per_layer)
        return self.params_list
