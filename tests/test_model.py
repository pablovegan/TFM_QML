import unittest

import numpy as np
from numpy.testing import assert_allclose
import pennylane as qml
from scipy.optimize import check_grad

from qubit_approximant import Model


@qml.qnode(qml.device("default.qubit", wires=1))
def circuit(x, θ, w) -> np.ndarray:
    for i in range(w.size):
        qml.RX(x * w[i] + θ[0, i], wires=0)
        qml.RY(θ[1, i], wires=0)
        qml.RZ(θ[2, i], wires=0)
    return qml.state()


class TestModel(unittest.TestCase):
    """Testing our modules."""

    def setUp(self) -> None:
        self.x = np.linspace(-2, 2, 100)
        layers = np.random.randint(1, 12)
        self.θ = np.random.randn(3 * layers).reshape(3, layers)
        self.w = np.random.randn(layers)

    def test_encoding(self):
        pennylane_list = []
        for x in self.x:
            pennylane_list.append(circuit(x, self.θ, self.w)[0])
        pennylane_list = np.array(pennylane_list)

        model = Model(x=self.x, encoding="amp")
        assert_allclose(
            model(self.θ, self.w),
            pennylane_list,
            rtol=1e-6,
            atol=1e-7,
            err_msg="Amplitude encoding not working.",
        )

    def test_grad_layer(self):
        model = Model(x=self.x, encoding="amp")
        δ = 0.000001
        w = 2
        θ0 = np.random.randn(3)
        θ1 = θ0.copy()
        θ1[0] += δ
        DUx_approx = (model._layer(θ1, w) - model._layer(θ0, w)) / δ
        DUx = model._grad_layer(θ0, w)[1]
        assert_allclose(DUx_approx, DUx, rtol=1e-5, atol=1e-6)

    def test_grad_prob_encoding(self):
        model = Model(x=self.x, encoding="prob")
        φ = 0.3 * np.random.randn(4 * 6)

        def split(φ):
            layers = φ.size // 4
            return φ[0:layers], φ[layers:].reshape(3, layers)

        def fun(φ):
            w, θ = split(φ)
            return np.sum(model(θ, w))

        def grad(φ):
            w, θ = split(φ)
            return np.sum(model._grad_prob(θ, w)[0], axis=0)

        assert check_grad(fun, grad, φ) < 5e-5, f"Check_grad = {check_grad(fun, grad, φ)}"


if __name__ == "__main__":
    unittest.main()
