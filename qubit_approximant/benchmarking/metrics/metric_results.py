from typing import Callable

import numpy as np

from qubit_approximant import Model
from .metrics import l2_norm, l1_norm, inf_norm, infidelity


def metric_results(
    params_list: list[np.ndarray],
    model: Model,
    fn: Callable,
    fn_kwargs: dict,
    x_limits: tuple[float, float],
):
    """Returns four lists of errors, one for each layer."""
    l1_list = []
    l2_list = []
    inf_list = []
    infidelity_list = []

    def fn_eval(x):
        return fn(x, **fn_kwargs)

    for params in params_list:

        def fn_approx_eval(x):
            model.x = np.array([x])
            return model(params)[0]  # one element array

        l1_list.append(l1_norm(fn_eval, fn_approx_eval, x_limits))
        l2_list.append(l2_norm(fn_eval, fn_approx_eval, x_limits))
        infidelity_list.append(infidelity(fn_eval, fn_approx_eval, x_limits))

        def fn_approx_inf_eval(x):
            model.x = np.array(x)
            return model(params)[0]  # one element array

        inf_list.append(inf_norm(fn_eval, fn_approx_inf_eval, x_limits))

    return l1_list, l2_list, inf_list, infidelity_list
