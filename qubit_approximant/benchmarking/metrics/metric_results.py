from typing import Callable

import numpy as np

from qubit_approximant import Cost, Model, MultilayerOptimizer
from qubit_approximant.benchmark_metric import l2_norm, l1_norm, inf_norm, infidelity


def metric_results(
    fn: Callable,
    fn_kwargs: dict,
    x_limits: tuple[float, float],
    model: Model,
    cost: Cost,
    optimizer: MultilayerOptimizer,
    optimizer_kwargs: dict,
):

    params_list = optimizer(cost, cost.grad, **optimizer_kwargs)
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
        inf_list.append(inf_norm(fn_eval, fn_approx_eval, x_limits))
        infidelity_list.append(infidelity(fn_eval, fn_approx_eval, x_limits))

    return l1_list, l2_list, inf_list, infidelity_list
