import unittest

import numpy as np
from scipy.optimize import check_grad

from qubit_approximant import Model, Metric, Cost


class TestCost(unittest.TestCase):
    """Testing our modules."""

    def setUp(self) -> None:
        self.x = np.linspace(-2, 2, 100)
        self.fn = np.exp(-((self.x) ** 2) / (2 * 0.5**2)) / (0.5 * np.sqrt(2 * np.pi))
        np.random.seed(2)
        self.φ = 0.3 * np.random.randn(4 * 6)

    def test_grad_amp_mse(self):

        model = Model(x=self.x, encoding="amp")
        metric = Metric("MSE")
        cost = Cost(self.fn, model, metric)

        assert check_grad(cost, cost.grad, self.φ) < 1e-5, (
            f"Check_grad = {check_grad(cost, cost.grad, self.φ)}")

    def test_grad_amp_rmse(self):

        model = Model(x=self.x, encoding="amp")
        metric = Metric("RMSE")
        cost = Cost(self.fn, model, metric)

        assert check_grad(cost, cost.grad, self.φ) < 1e-5, (
            f"Check_grad = {check_grad(cost, cost.grad, self.φ)}")

    def test_grad_prob_mse(self):

        model = Model(x=self.x, encoding="prob")
        metric = Metric("MSE")
        cost = Cost(self.fn, model, metric)

        assert check_grad(cost, cost.grad, self.φ) < 1e-5, (
            f"Check_grad = {check_grad(cost, cost.grad, self.φ)}")

    def test_grad_prob_rmse(self):

        model = Model(x=self.x, encoding="prob")
        metric = Metric("RMSE")
        cost = Cost(self.fn, model, metric)

        assert check_grad(cost, cost.grad, self.φ) < 1e-5, (
            f"Check_grad = {check_grad(cost, cost.grad, self.φ)}")


if __name__ == "__main__":
    unittest.main()
