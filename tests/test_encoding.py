import unittest

from numpy import linspace, ndarray, array
import numpy as np
from numpy.testing import assert_allclose
from numpy.random import uniform, randint
from scipy.optimize import check_grad
import pennylane as qml

from cost import Cost


@qml.qnode(qml.device("default.qubit", wires=1))
def circuit(x, θ, w) -> ndarray:
    for i in range(w.size):
        qml.RX(x * w[i] + θ[0, i], wires=0)
        qml.RY(θ[1, i], wires=0)
        qml.RZ(θ[2, i], wires=0)
    return qml.state()


def split(φ):
    layers = φ.size // 4
    return φ[0:layers], φ[layers:].reshape(3, layers)


class TestEncoding(unittest.TestCase):
    """Testing our modules."""

    def setUp(self) -> None:
        self.x = uniform(-20, 20, 100)
        layers = randint(1, 12)
        self.φ = np.random.randn(layers * 4)
        self.θ = uniform(-5, 5, 3 * layers).reshape(3, layers)
        self.w = uniform(-6, 6, layers)

    def test_model(self):
        pennylane_list = []
        for x in self.x:
            pennylane_list.append(circuit(x, self.θ, self.w)[0])
        pennylane_list = array(pennylane_list)
        model = Cost(x=self.x, fn=0, encoding="amp")
        assert_allclose(
            model._encoding(self.θ, self.w),
            pennylane_list,
            rtol=1e-6,
            atol=1e-7,
            err_msg="Amplitude encoding not working.",
        )

    def test_grad_layer(self):
        model = Cost(x=self.x, fn=0, encoding="amp")
        δ = 0.000001
        w = 2
        θ0 = np.random.randn(3)
        θ1 = θ0.copy()
        θ1[0] += δ
        DUx_approx = (model._layer(θ1, w) - model._layer(θ0, w)) / δ
        DUx = model._der_layer(θ0, w)[1]
        assert_allclose(DUx_approx, DUx, rtol=1e-6, atol=1e-7)

    def test_grad_amp_encoding(self):
        model = Cost(x=self.x, fn=0, encoding="amp")
        φ = np.random.randn(4 * 6)

        def fun(φ):
            w, θ = split(φ)
            return np.sum(model._encoding(θ, w))

        def grad(φ):
            w, θ = split(φ)
            return np.sum(model._grad_amp_encoding(θ, w)[0], axis=0)

        print("check grad ", check_grad(fun, grad, φ))
        assert check_grad(fun, grad, φ) < 1e-5

    def test_grad_prob_encoding(self):
        model = Cost(x=self.x, fn=0, encoding="prob")
        φ = np.random.randn(4 * 6)

        def fun(φ):
            w, θ = split(φ)
            return np.sum(model._encoding(θ, w))

        def grad(φ):
            w, θ = split(φ)
            return np.sum(model._grad_prob_encoding(θ, w)[0], axis=0)

        print("check grad ", check_grad(fun, grad, φ))
        assert check_grad(fun, grad, φ) < 1e-5

    def test_grad_mse(self):
        
        x = np.linspace(-2,2,100)
        fn = np.exp(-(x)**2/(2*0.5**2)) / (0.5*np.sqrt(2*np.pi)) 
        model = Cost(x=x, fn=fn, encoding="prob")
        φ = np.random.randn(4 * 6)

        def fun(φ):
            w, θ = split(φ)
            fn = model._encoding(θ, w)
            return model.mse_error(model.fn, fn)

        def grad(φ):
            w, θ = split(φ)
            return model.grad_mse(θ, w)

        print("check grad ", check_grad(fun, grad, φ))
        assert check_grad(fun, grad, φ) < 1e-2

if __name__ == "__main__":
    unittest.main()
