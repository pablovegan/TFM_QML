"""Benchmark our encoding and optimizer for multiple seeds."""

import pickle
from typing import Callable, Tuple

import numpy as np
import dill
from pathos.multiprocessing import ProcessingPool, cpu_count
from mpi4py import MPI

# from multiprocessing import Pool, cpu_count

from qubit_approximant.core import Circuit, Cost, MultilayerOptimizer
from qubit_approximant.benchmarking import metric_results


def benchmark_seeds(
    num_seeds: int,
    fn: Callable,
    fn_kwargs: dict,
    circuit: Circuit,
    cost: Cost,
    optimizer: MultilayerOptimizer,
    filename: str,
) -> None:
    """Benchmark a circuit ansatz, cost and multilayer optimizer for different seeds.
    The benchmarks are paralelized throught different nodes and cpu cores. The results
    are saved in a file in the order:
    layers_list, l1_list, l2_list, inf_list, infidelity_list;
    where each list contains the metric for each seed with every number of layers.

    Parameters
    ----------
    num_seeds : int
        Number of seeds to benchmark.
    fn : Callable
        Function we want to approximate.
    fn_kwargs : dict
        Keyword arguments for 'fn'.
    circuit : Circuit
        Circuit ansatz used to encode the function.
    cost : Cost
        Cost or loss function used in the optimization to estimate the error
        in approximating fn using the quantum circuit.
    optimizer : MultilayerOptimizer
        Optimizer to obtain optimum parameters for multiple number of layers.
    filename : str
        File name to store the results (they may take some time).
    """
    opt = optimizer
    num_layer = opt.max_layer - opt.min_layer + 1

    print("Comienzan los cálculos.")

    # dill allows to 'pickle' more complex objects
    MPI.pickle.__init__(dill.dumps, dill.loads)  # type: ignore[misc]
    comm = MPI.COMM_WORLD
    num_nodes = comm.Get_size()
    rank = comm.Get_rank()
    seeds_node = num_seeds // num_nodes
    rng = np.random.default_rng()
    seed_list = rng.choice(range(rank * 2000, (rank + 1) * 2000), seeds_node, replace=False)

    def metric_results_seed(seed: int) -> Tuple[list[float], ...]:
        rng = np.random.default_rng(seed)
        params = opt.new_layer_coef * rng.standard_normal(circuit.params_layer * opt.min_layer)
        params_list = opt(cost, cost.grad, params)
        return metric_results(fn, fn_kwargs, circuit, params_list)

    with ProcessingPool(cpu_count()) as p:
        metrics_per_seed = p.map(metric_results_seed, seed_list)

    print("Cálculos terminados.")

    l1_seeds = [metrics[0] for metrics in metrics_per_seed]
    l2_seeds = [metrics[1] for metrics in metrics_per_seed]
    inf_seeds = [metrics[2] for metrics in metrics_per_seed]
    infidelity_seeds = [metrics[3] for metrics in metrics_per_seed]

    metrics_send = np.array([l1_seeds, l2_seeds, inf_seeds, infidelity_seeds])
    metrics_receive = None

    if rank == 0:
        # Receive data from the resting nodes
        metrics_receive = np.zeros((num_nodes, len(metrics_send), seeds_node, num_layer))

    # comm.Barrier()   # wait for everybody to synchronize _here_
    comm.Gather(metrics_send, metrics_receive, root=0)  # Node 0 receives data
    # comm.Barrier()

    if rank == 0:
        # dim results (nodes)x(metrics)x(seeds_node)x(layers)
        metrics_receive = np.swapaxes(metrics_receive, 1, 2)  # type: ignore
        # dim results (nodes)x(seeds_node)x(metrics)x(layers)
        metrics = np.concatenate(metrics_receive, axis=0)
        # dim results (seeds)x(metrics)x(layers)
        metrics = np.swapaxes(metrics, 0, 1)
        # dim results (metrics)x(seeds)x(layers)

        layer_list = list(range(opt.min_layer, opt.max_layer + 1))

        with open(filename + ".pkl", "wb") as file:
            pickle.dump(
                (layer_list, metrics[0, ...], metrics[1, ...], metrics[2, ...], metrics[3, ...]),
                file,
            )
