import unittest

import numpy as np

from qubit_approximant import RotationsModel, Cost, BlackBoxOptimizer, AdamOptimizer
from qubit_approximant.core.optimizer import IncrementalOptimizer, NonIncrementalOptimizer


class TestOptimizer(unittest.TestCase):
    """Testing our optimizer."""

    def setUp(self) -> None:
        self.x = np.linspace(-2, 2, 100)
        self.fn = np.exp(-((self.x) ** 2) / (2 * 0.5**2)) / (0.5 * np.sqrt(2 * np.pi))
        layers = 8
        np.random.seed(20)
        self.params = 0.7 * np.random.randn(4 * layers)

    def test_blackbox(self):
        model = RotationsModel(x=self.x, encoding="prob")
        cost = Cost(self.fn, model, metric="mse")
        opt = BlackBoxOptimizer(method="L-BFGS-B")
        params = opt(cost, cost.grad, self.params)
        assert cost(params) < 1e-5

    def test_adam(self):
        model = RotationsModel(x=self.x, encoding="prob")
        cost = Cost(self.fn, model, metric="mse")
        opt = AdamOptimizer(5000)
        params = opt(cost, cost.grad, self.params)
        assert cost(params) < 1e-2  # Adam optimizer is not working very good :(


class TestIncrementalOptimizer(unittest.TestCase):
    """Testing our optimizer."""

    def setUp(self) -> None:
        self.x = np.linspace(-2, 2, 100)
        self.fn = np.exp(-((self.x) ** 2) / (2 * 0.5**2)) / (0.5 * np.sqrt(2 * np.pi))
        self.min_layers = 6
        self.max_layers = 9
        np.random.seed(20)
        self.params = 0.3 * np.random.randn(4 * self.min_layers)

    def test_blackbox(self):
        model = RotationsModel(x=self.x, encoding="prob")
        cost = Cost(self.fn, model, metric="mse")
        opt = BlackBoxOptimizer(method="L-BFGS-B")
        multilayer_opt = IncrementalOptimizer(self.min_layers, self.max_layers, opt, "final", 0.3)
        params_list = multilayer_opt(cost, cost.grad, self.params)
        for i, params in enumerate(params_list):
            assert cost(params) < 1e-4, f"Error in layer {i} with cost = {cost(params)}"

        mean_diff, std_diff = multilayer_opt.inital_params_diff


class TestNonIncrementalOptimizer(unittest.TestCase):
    """Testing our optimizer."""

    def setUp(self) -> None:
        self.x = np.linspace(-2, 2, 100)
        self.fn = np.exp(-((self.x) ** 2) / (2 * 0.5**2)) / (0.5 * np.sqrt(2 * np.pi))
        self.min_layers = 6
        self.max_layers = 9
        np.random.seed(20)
        self.params = 0.3 * np.random.randn(4 * self.min_layers)

    def test_blackbox(self):
        model = RotationsModel(x=self.x, encoding="prob")
        cost = Cost(self.fn, model, metric="mse")
        opt = BlackBoxOptimizer(method="L-BFGS-B")
        multilayer_opt = NonIncrementalOptimizer(self.min_layers, self.max_layers, opt, 0.3)
        params_list = multilayer_opt(cost, cost.grad, self.params)
        for i, params in enumerate(params_list):
            assert cost(params) < 1e-4, f"Error in layer {i} with cost = {cost(params)}"


if __name__ == "__main__":
    unittest.main()
