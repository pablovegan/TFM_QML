import unittest

import numpy as np
from numpy.testing import assert_allclose
import pennylane as qml

from cost import Cost


@qml.qnode(qml.device("default.qubit", wires=1))
def circuit(x, θ, w) -> np.ndarray:
    for i in range(w.size):
        qml.RX(x * w[i] + θ[0, i], wires=0)
        qml.RY(θ[1, i], wires=0)
        qml.RZ(θ[2, i], wires=0)
    return qml.state()


class TestEncoding(unittest.TestCase):
    """Testing our modules."""

    def setUp(self) -> None:
        self.x = np.linspace(-20, 20, 100)
        layers = np.random.randint(1, 12)
        self.θ = np.random.randn(3 * layers).reshape(3, layers)
        self.w = np.random.randn(layers)

    def test_model(self):
        pennylane_list = []
        for x in self.x:
            pennylane_list.append(circuit(x, self.θ, self.w)[0])
        pennylane_list = np.array(pennylane_list)

        model = Cost(x=self.x, fn=0, encoding="amp")
        assert_allclose(
            model.encoding(self.θ, self.w),
            pennylane_list,
            rtol=1e-6,
            atol=1e-7,
            err_msg="Amplitude encoding not working.",
        )


if __name__ == "__main__":
    unittest.main()
