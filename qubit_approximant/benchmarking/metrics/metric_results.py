from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray

from qubit_approximant import Circuit
from .metrics import l2_norm, l1_norm, inf_norm, infidelity


def metric_results(
    fn: Callable, fn_kwargs: dict, circuit: Circuit, params_list: list[NDArray]
) -> Tuple[list[float], ...]:
    """Returns 4 lists of error metrics, one for each layer. The metrics are:
        - L1 norm
        - L2 norm
        - Infinity norm
        - Infidelity

    Parameters
    ----------
    fn : Callable
        Function we want to approximate.
    fn_kwargs : dict
        Keyword arguments for 'fn'.
    circuit : Circuit
        Circuit used to model 'fn'.
    params_list : list[NDArray]
        List of parameters for the circuit with different number of layers.

    Returns
    -------
    Tuple[list[float], ...]
        Returns lists of L1 norms, L2 norms, infinity norms and infidelities
        for every number of layers.
    """
    l1_list = []
    l2_list = []
    inf_list = []
    infidelity_list = []

    save_model_x = circuit.x.copy()  # copy just in case, although not necessary
    x_limits = (circuit.x[0], circuit.x[-1])

    def fn_eval(x):
        return fn(x, **fn_kwargs)

    for params in params_list:

        def fn_approx_eval(x):
            circuit.x = np.array([x])
            return circuit.encoding(params)[0]  # one element array

        l1_list.append(l1_norm(fn_eval, fn_approx_eval, x_limits))
        l2_list.append(l2_norm(fn_eval, fn_approx_eval, x_limits))
        infidelity_list.append(infidelity(fn_eval, fn_approx_eval, x_limits))

        def fn_approx_inf_eval(x):
            circuit.x = np.array(x)
            return circuit.encoding(params)  # one element array

        inf_list.append(inf_norm(fn_eval, fn_approx_inf_eval, x_limits))

    circuit.x = save_model_x

    return l1_list, l2_list, inf_list, infidelity_list
