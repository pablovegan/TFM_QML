import unittest

import numpy as np
from scipy.optimize import check_grad

from qubit_approximant import CircuitRxRyRz, Cost


class TestCost(unittest.TestCase):
    """Testing our modules."""

    def setUp(self) -> None:
        self.x = np.linspace(-2, 2, 100)
        self.fn = np.exp(-((self.x) ** 2) / (2 * 0.5**2)) / (0.5 * np.sqrt(2 * np.pi))
        np.random.seed(2)
        self.φ = 0.3 * np.random.randn(4 * 6)

    def test_grad_amp_mse(self):

        circuit = CircuitRxRyRz(x=self.x, encoding="amp")
        cost = Cost(self.fn, circuit, metric_str="mse")

        assert check_grad(cost, cost.grad, self.φ) < 1e-5, (
            f"Check_grad = {check_grad(cost, cost.grad, self.φ)}")

    def test_grad_amp_rmse(self):

        circuit = CircuitRxRyRz(x=self.x, encoding="amp")
        cost = Cost(self.fn, circuit, metric_str="rmse")

        assert check_grad(cost, cost.grad, self.φ) < 1e-5, (
            f"Check_grad = {check_grad(cost, cost.grad, self.φ)}")

    def test_grad_amp_mse_weighted(self):

        circuit = CircuitRxRyRz(x=self.x, encoding="amp")
        cost = Cost(self.fn, circuit, metric_str="mse_weighted")

        assert check_grad(cost, cost.grad, self.φ) < 1e-5, (
            f"Check_grad = {check_grad(cost, cost.grad, self.φ)}")

    def test_grad_prob_mse(self):

        circuit = CircuitRxRyRz(x=self.x, encoding="prob")
        cost = Cost(self.fn, circuit, metric_str="mse")

        assert check_grad(cost, cost.grad, self.φ) < 1e-5, (
            f"Check_grad = {check_grad(cost, cost.grad, self.φ)}")

    def test_grad_prob_rmse(self):

        circuit = CircuitRxRyRz(x=self.x, encoding="prob")
        cost = Cost(self.fn, circuit, metric_str="rmse")

        assert check_grad(cost, cost.grad, self.φ) < 1e-5, (
            f"Check_grad = {check_grad(cost, cost.grad, self.φ)}")

    def test_grad_prob_mse_weighted(self):

        circuit = CircuitRxRyRz(x=self.x, encoding="prob")
        cost = Cost(self.fn, circuit, metric_str="mse_weighted")

        assert check_grad(cost, cost.grad, self.φ) < 1e-5, (
            f"Check_grad = {check_grad(cost, cost.grad, self.φ)}")

    def test_grad_prob_log_cosh(self):

        circuit = CircuitRxRyRz(x=self.x, encoding="prob")
        cost = Cost(self.fn, circuit, metric_str="log_cosh")

        assert check_grad(cost, cost.grad, self.φ) < 1e-5, (
            f"Check_grad = {check_grad(cost, cost.grad, self.φ)}")

    def test_grad_prob_kl_divergence(self):

        circuit = CircuitRxRyRz(x=self.x, encoding="prob")
        cost = Cost(self.fn, circuit, metric_str="kl_divergence")

        assert check_grad(cost, cost.grad, self.φ) < 1e-5, (
            f"Check_grad = {check_grad(cost, cost.grad, self.φ)}")


if __name__ == "__main__":
    unittest.main()
