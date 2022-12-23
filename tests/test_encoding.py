import unittest

from numpy import linspace, ndarray, array
from numpy.testing import assert_allclose
from numpy.random import uniform, randint
from scipy.optimize import check_grad
import pennylane as qml

from cost import Model


@qml.qnode(qml.device('default.qubit', wires = 1))
def circuit(x, θ, w) -> ndarray:
    for i in range(w.size):
        qml.RX(x * w[i] + θ[0, i], wires = 0)
        qml.RY(θ[1, i], wires = 0)
        qml.RZ(θ[2, i], wires = 0)
    return qml.state()


class TestEncoding(unittest.TestCase):
    """Testing our modules."""
    
    def setUp(self) -> None:
        self.x = uniform(-20, 20, 1000)
        layers = randint(1, 12)
        self.θ = uniform(-5, 5, 3 * layers).reshape(3, layers)
        self.w = uniform(-6, 6, layers)


    def test_model(self):
        pennylane_list = []
        for x in self.x:
            pennylane_list.append(circuit(x, self.θ, self.w)[0]) # pylint: disable=unsubscriptable-object
        pennylane_list = array(pennylane_list)
        model = Model('amp')
        assert_allclose(model.evaluate(self.x, self.θ, self.w),
                        pennylane_list,
                        rtol = 1e-6,
                        atol = 1e-7,
                        err_msg = "Amplitude encoding not working.")
        
    def test_grad(self):
        model = Model('amp')
        assert check_grad(model.evaluate, model.grad, self.x, self.θ, self.w) < 1e-6
        
        model = Model('prob')
        assert check_grad(model.evaluate, model.grad, self.x, self.θ, self.w) < 1e-6
                          

if __name__ == '__main__':
    unittest.main()
