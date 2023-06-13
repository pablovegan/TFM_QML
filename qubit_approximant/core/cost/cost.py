"""Cost function to use in our optimizer."""

from numpy.typing import NDArray

from qubit_approximant.core import Circuit
from ._cost_metrics import *  # noqa


class Cost:
    """
    Create a cost function from the encoding and the metric.

    Attributes
    ----------
    metric: Callable
        The metric or loss function to quantify how well our circuit
        approximates the target function.
    grad_metric: Callable
        The gradient of the metric or loss function.
    circuit: Circuit
        Quantum circuit that encodes our function.
    fn: NDArray
        Function we desire to approximate.
    """

    def __init__(self, fn: NDArray, circuit: Circuit, metric: str) -> None:
        """
        Parameters
        ----------
        fn : NDArray
            Function we desire to approximate.
        circuit : Circuit
            Quantum circuit that encodes our function.
        metric : str
            Name of the metric we want to use.
            Allowed values are:
                - 'mse' (mean square error)
                - 'rmse' (root mean square error)
                - 'mse_weighted' (mse weighted by fn)
                - 'kl_divergence'
                - 'log_cosh'.
        """
        try:
            self.metric = globals()[metric]
            self.grad_metric = globals()["grad_" + metric]
        except KeyError as e:
            raise ValueError("Invalid metric '{metric}'. Choose between 'MSE' or 'RMSE'.") from e

        self.circuit = circuit
        self.fn = fn

    def __call__(self, params: NDArray) -> float:
        """Evaluate the cost function given the parameters of the circuit.

        Parameters
        ----------
        params : NDArray
            Parameters of the quantum gates in the layer.

        Returns
        -------
        float
            The value of the cost function for the chosen circuit and metric.
        """
        fn_approx = self.circuit.encoding(params)
        return self.metric(self.fn, fn_approx)

    def grad(self, params: NDArray) -> NDArray:
        """Return the gradient of the cost function.

        Parameters
        ----------
        params : NDArray
            Parameters of the quantum gates in the layer.

        Returns
        -------
        NDArray
            Gradient of the cost.
        """
        grad_fn_approx, fn_approx = self.circuit.grad_encoding(params)
        return self.grad_metric(self.fn, fn_approx, grad_fn_approx)
