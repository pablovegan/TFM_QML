"""Cost function to use in our optimizer."""

from numpy import ndarray

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
    fn: ndarray
        Function we desire to approximate.
    """

    def __init__(self, fn: ndarray, circuit: Circuit, metric_str: str) -> None:
        """
        Parameters
        ----------
        fn : ndarray
            Function we desire to approximate.
        circuit : Circuit
            Quantum circuit that encodes our function.
        metric_str : str
            Name of the metric we want to use.
            Allowed values are:
                - 'mse' (mean square error)
                - 'rmse' (root mean square error)
                - 'mse_weighted' (mse weighted by fn)
                - 'kl_divergence'
                - 'log_cosh'.
        """
        try:
            self.metric = globals()[metric_str]
            self.grad_metric = globals()["grad_" + metric_str]
        except KeyError:
            raise ValueError("Invalid metric '{metric}'. Choose between 'MSE' or 'RMSE'.")

        self.circuit = circuit
        self.fn = fn

    def __call__(self, params: ndarray) -> float:
        """Evaluate the cost function given the parameters of the circuit.

        Parameters
        ----------
        params : ndarray
            Parameters of the quantum gates in the layer.

        Returns
        -------
        float
            The value of the cost function for the chosen circuit and metric.
        """
        fn_approx = self.circuit.encoding(params)
        return self.metric(self.fn, fn_approx)

    def grad(self, params: ndarray) -> ndarray:
        """Return the gradient of the cost function.

        Parameters
        ----------
        params : ndarray
            Parameters of the quantum gates in the layer.

        Returns
        -------
        ndarray
            Gradient of the cost.
        """
        grad_fn_approx, fn_approx = self.circuit.grad_encoding(params)
        return self.grad_metric(self.fn, fn_approx, grad_fn_approx)
