import unittest

import numpy as np
from numpy.testing import assert_allclose
import pennylane as qml
from scipy.optimize import check_grad

from qubit_approximant import CircuitRxRyRz, CircuitRxRy, CircuitRy


class TestCircuitRxRyRz(unittest.TestCase):
    """Testing our modules."""

    def setUp(self) -> None:
        self.x = np.linspace(-2, 2, 100)
        layers = np.random.randint(1, 12)
        self.params = np.random.randn(4 * layers)

    def test_encoding(self):

        def pennylane_circuit(x, params):
            params = params.reshape(-1, 4)

            @qml.qnode(qml.device("default.qubit", wires=1))
            def _circuit() -> np.ndarray:
                for i in range(params.shape[0]):
                    qml.RX(x * params[i, 0] + params[i, 1], wires=0)
                    qml.RY(params[i, 2], wires=0)
                    qml.RZ(params[i, 3], wires=0)
                return qml.state()
            return _circuit()

        pennylane_list = []
        for x in self.x:
            pennylane_list.append(pennylane_circuit(x, self.params)[0])
        pennylane_list = np.array(pennylane_list)

        circuit = CircuitRxRyRz(x=self.x, encoding="amp")
        assert_allclose(
            circuit.encoding(self.params),
            pennylane_list,
            rtol=1e-6,
            atol=1e-7,
            err_msg="Amplitude encoding not working.",
        )

    def test_grad_layer(self):
        circuit = CircuitRxRyRz(x=self.x, encoding="amp")
        δ = 0.000001
        params0 = np.random.randn(4)
        params1 = params0.copy()
        params1[1] += δ
        DUx_approx = (circuit.layer(params1) - circuit.layer(params0)) / δ
        DUx = circuit.grad_layer(params0)[1]
        assert_allclose(DUx_approx, DUx, rtol=1e-5, atol=1e-6)

    def test_grad_prob_encoding(self):
        circuit = CircuitRxRyRz(x=self.x, encoding="prob")

        def fun(params):
            return np.sum(circuit.encoding(params))

        def grad(params):
            return np.sum(circuit._grad_prob(params)[0], axis=0)

        assert check_grad(fun, grad, self.params) < 5e-5, f"Check_grad = {check_grad(fun, grad, self.params)}"


class TestCircuitRxRy(unittest.TestCase):
    """Testing our modules."""

    def setUp(self) -> None:
        self.x = np.linspace(-2, 2, 100)
        layers = np.random.randint(1, 12)
        self.params = 0.3 * np.random.randn(3 * layers)

    def test_grad_layer(self):
        circuit = CircuitRxRy(x=self.x, encoding="amp")
        δ = 0.000001
        params0 = np.random.randn(3)
        params1 = params0.copy()
        params1[1] += δ
        DUx_approx = (circuit.layer(params1) - circuit.layer(params0)) / δ
        DUx = circuit.grad_layer(params0)[1]
        assert_allclose(DUx_approx, DUx, rtol=1e-5, atol=1e-6)

    def test_grad_prob_encoding(self):
        circuit = CircuitRxRy(x=self.x, encoding="prob")

        def fun(params):
            return np.sum(circuit.encoding(params))

        def grad(params):
            return np.sum(circuit._grad_prob(params)[0], axis=0)

        assert check_grad(fun, grad, self.params) < 5e-5, f"Check_grad = {check_grad(fun, grad, self.params)}"


class TestCircuitRy(unittest.TestCase):
    """Testing our modules."""

    def setUp(self) -> None:
        self.x = np.linspace(-2, 2, 100)
        layers = np.random.randint(1, 12)
        self.params = 0.3 * np.random.randn(2 * layers)

    def test_grad_layer(self):
        circuit = CircuitRy(x=self.x, encoding="amp")
        δ = 0.000001
        params0 = np.random.randn(2)
        params1 = params0.copy()
        params1[1] += δ
        DUx_approx = (circuit.layer(params1) - circuit.layer(params0)) / δ
        DUx = circuit.grad_layer(params0)[1]
        assert_allclose(DUx_approx, DUx, rtol=1e-5, atol=1e-6)

    def test_grad_prob_encoding(self):
        circuit = CircuitRy(x=self.x, encoding="prob")

        def fun(params):
            return np.sum(circuit.encoding(params))

        def grad(params):
            return np.sum(circuit._grad_prob(params)[0], axis=0)

        assert check_grad(fun, grad, self.params) < 5e-5, f"Check_grad = {check_grad(fun, grad, self.params)}"


if __name__ == "__main__":
    unittest.main()
