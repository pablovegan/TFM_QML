import pytest
from typing import Type

import numpy as np
from numpy.testing import assert_allclose
import pennylane as qml
from scipy.optimize import check_grad

from qubit_approximant import Circuit, CircuitRxRyRz, CircuitRxRy, CircuitRy


x = np.linspace(-2, 2, 100)
layers = np.random.randint(1, 12)
params = np.random.randn(4 * layers)


@pytest.mark.parametrize(("x", "params"), ((x, params),))
def test_encoding_CircuitRxRyRz(x: np.ndarray, params: np.ndarray):
    def pennylane_circuit(x_point, params):
        params = params.reshape(-1, 4)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def _circuit() -> np.ndarray:
            for i in range(params.shape[0]):
                qml.RX(x_point * params[i, 0] + params[i, 1], wires=0)
                qml.RY(params[i, 2], wires=0)
                qml.RZ(params[i, 3], wires=0)
            return qml.state()

        return _circuit()

    pennylane_list = []
    for x_point in x:
        pennylane_list.append(pennylane_circuit(x_point, params)[0])
    pennylane_list = np.array(pennylane_list)  # type: ignore

    circuit = CircuitRxRyRz(x=x, encoding="amp")
    assert_allclose(
        circuit.encoding(params),
        pennylane_list,
        rtol=1e-6,
        atol=1e-7,
        err_msg="Amplitude encoding not working.",
    )


@pytest.mark.parametrize(
    ("circuit_class", "x", "params"),
    (
        (CircuitRxRyRz, x, np.random.randn(4)),
        (CircuitRxRy, x, np.random.randn(3)),
        (CircuitRy, x, np.random.randn(2)),
    ),
)
def test_grad_layer(circuit_class: Type[Circuit], x: np.ndarray, params: np.ndarray):
    circuit = circuit_class(x=x, encoding="amp")  # type: ignore
    δ = 0.000001
    params_shifted = params.copy()
    params_shifted[1] += δ
    DUx_approx = (circuit.layer(params_shifted) - circuit.layer(params)) / δ
    DUx = circuit.grad_layer(params)[1]
    assert_allclose(DUx_approx, DUx, rtol=1e-5, atol=1e-6)


layers = np.random.randint(1, 12)


@pytest.mark.parametrize(
    ("circuit_class", "x", "params"),
    (
        (CircuitRxRyRz, x, np.random.randn(4 * layers)),
        (CircuitRxRy, x, np.random.randn(3 * layers)),
        (CircuitRy, x, np.random.randn(2 * layers)),
    ),
)
def test_grad_prob_encoding(circuit_class: Type[Circuit], x: np.ndarray, params: np.ndarray):
    circuit = circuit_class(x=x, encoding="prob")  # type: ignore

    def fun(params):
        return np.sum(circuit.encoding(params))

    def grad(params):
        return np.sum(circuit.grad_prob(params)[0], axis=0)

    assert check_grad(fun, grad, params) < 5e-5, f"Check_grad = {check_grad(fun, grad, params)}"
