import unittest

import numpy as np

from qubit_approximant import RotationsModel, Cost, Metric, BlackBoxOptimizer, AdamOptimizer


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
        metric = Metric("mse")
        cost = Cost(self.fn, model, metric)
        opt = BlackBoxOptimizer(method="L-BFGS-B")
        params = opt(cost, cost.grad, self.params)
        fn_approx = model(params)
        assert metric(fn_approx - self.fn) < 1e-5

    def test_adam(self):
        model = RotationsModel(x=self.x, encoding="prob")
        metric = Metric("mse")
        cost = Cost(self.fn, model, metric)
        opt = AdamOptimizer(5000)
        params = opt(cost, cost.grad, self.params)
        fn_approx = model(params)
        assert metric(fn_approx - self.fn) < 1e-2  # Adam optimizer is not working very good :(


if __name__ == "__main__":
    unittest.main()
