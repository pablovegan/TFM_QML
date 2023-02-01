import unittest

import numpy as np
from numpy.testing import assert_allclose
from numpy.random import randint
from scipy.optimize import check_grad

from model import Model, Cost


def split(φ):
    layers = φ.size // 4
    return φ[0:layers], φ[layers:].reshape(3, layers)


class TestGradient(unittest.TestCase):
    """Testing our modules."""

    def setUp(self) -> None:
        self.x = np.linspace(-2, 2, 100)
        # self.x = np.array([2])
        self.fn = np.exp(-((self.x) ** 2) / (2 * 0.5**2)) / (0.5 * np.sqrt(2 * np.pi))
        layers = randint(1, 12)
        self.φ = np.random.randn(layers * 4)

    def test_grad_amp_mse(self):

        model = Model(x=self.x, fn=self.fn, encoding="amp")
        np.random.seed(2)
        φ = 0.3 * np.random.randn(4 * 6)

        def fun(φ):
            w, θ = split(φ)
            fn = model.encoding(θ, w)
            return model.mse_error(model.fn, fn)

        def grad(φ):
            w, θ = split(φ)
            return model.grad_mse(θ, w)

        assert check_grad(fun, grad, φ) < 1e-5, f"Check_grad = {check_grad(fun, grad, φ)}"

    def test_grad_amp_rmse(self):

        model = Model(x=self.x, fn=self.fn, encoding="amp")
        np.random.seed(2)
        φ = 0.3 * np.random.randn(4 * 6)

        def fun(φ):
            w, θ = split(φ)
            fn = model.encoding(θ, w)
            return model.rmse_error(model.fn, fn)

        def grad(φ):
            w, θ = split(φ)
            return model.grad_rmse(θ, w)

        assert check_grad(fun, grad, φ) < 1e-5, f"Check_grad = {check_grad(fun, grad, φ)}"

    def test_grad_prob_mse(self):

        model = Model(x=self.x, fn=self.fn, encoding="prob")
        np.random.seed(2)
        φ = 0.3 * np.random.randn(4 * 6)

        def fun(φ):
            w, θ = split(φ)
            fn = model.encoding(θ, w)
            return model.mse_error(model.fn, fn)

        def grad(φ):
            w, θ = split(φ)
            return model.grad_mse(θ, w)

        assert check_grad(fun, grad, φ) < 1e-5, f"Check_grad = {check_grad(fun, grad, φ)}"

    def test_grad_prob_rmse(self):

        model = Model(x=self.x, fn=self.fn, encoding="prob")
        np.random.seed(2)
        φ = 0.3 * np.random.randn(4 * 6)

        def fun(φ):
            w, θ = split(φ)
            fn = model.encoding(θ, w)
            return model.rmse_error(model.fn, fn)

        def grad(φ):
            w, θ = split(φ)
            return model.grad_rmse(θ, w)

        assert check_grad(fun, grad, φ) < 1e-5, f"Check_grad = {check_grad(fun, grad, φ)}"


if __name__ == "__main__":
    unittest.main()
