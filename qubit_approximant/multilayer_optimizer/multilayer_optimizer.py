"""Incremental optimizer"""

from typing import Callable
from abc import ABC, abstractmethod

import numpy as np

from qubit_approximant.optimizer import Optimizer, BlackBoxOptimizer, GDOptimizer, AdamOptimizer


class MultilayerOptimizer(ABC):
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
        self.min_layer = min_layer
        self.max_layer = max_layer

        if new_layer_position in MultilayerOptimizer.layer_positions:
            self.new_layer_position = new_layer_position
        else:
            raise ValueError(
                f"new_layer_position = {new_layer_position} is not supported. "
                "Try 'initial', 'middle', 'final' or 'random'"
            )

        if method in MultilayerOptimizer.blackbox_methods:
            self.optimizer: Optimizer = BlackBoxOptimizer(method, method_kwargs)
        elif method == "GD":
            self.optimizer = GDOptimizer(**method_kwargs)
        elif method == "Adam":
            self.optimizer = AdamOptimizer(**method_kwargs)
        else:
            raise ValueError(f"Optimization {method} is not supported.")

    @abstractmethod
    def __call__(
        self,
        cost: Callable,
        grad_cost: Callable,
        init_params: np.ndarray,
        new_layer_coef: float = 0.3,
    ) -> list[np.ndarray]:
        ...
