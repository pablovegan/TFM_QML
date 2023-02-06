import pickle
import numpy as np
from pathos.multiprocessing import ProcessingPool, cpu_count
from mpi4py import MPI
import dill
from multiprocessing import Pool, cpu_count

from qubit_approximant import Cost, Model, MultilayerOptimizer
from qubit_approximant.benchmark_metric import l2_norm, l1_norm, inf_norm, infidelity



def benchmark_seed(seeds, x, cost: Cost, optimizer: MultilayerOptimizer):
    # Computación paralela
    MPI.pickle.__init__(dill.dumps, dill.loads)
    comm = MPI.COMM_WORLD
    nodes = comm.Get_size()
    rank = comm.Get_rank()
    seeds_node = seeds // nodes

    seed_array = np.random.choice(
        range(rank * 2000, (rank + 1) * 2000), seeds_node, replace=False
    )

    # Computación multiprocesador
    with ProcessingPool(cpu_count()) as p:
        # l2_list, l1_list, inf_list, fid_list, cost_list, mean_diff, std_diff
        results = p.map(
            lambda seed: graficas_errores(
                seed,
                min_layers=min_layers,
                max_layers=max_layers,
                x=x,
                f=f,
                grid_size=grid_size,
                function=function,
                f_params=f_params,
                interval=interval,
                int_method=int_method,
                opt_method=opt_method,
                φ_init=φ_init,
                show_plot=show_plot,
                show_final_plot=show_final_plot,
                show_error_plot=show_error_plot,
                show_diff=show_diff,
                print_cost=print_cost,
                cost_fun=cost_fun,
                incremental_opt=incremental_opt,
                print_params=print_params,
                cc=cc,
                new_layer_position=new_layer_position,
                new_layer_coef=new_layer_coef,
                plot_cost_error=plot_cost_error,
                method_params=method_params,
                probability=probability,
                model=model,
            ),
            seed_array,
        )



def mean_seed_errores_parallel(
    min_layers,
    max_layers,
    x,
    f,
    opt_method: str = "L-BFGS-B",
    init_params: Optional[np.ndarray] = None,
    probability: Optional[bool] = None,
    show_plot: bool = False,
    show_final_plot: bool = True,
    show_error_plot: bool = True,
    show_diff=False,
    print_cost: bool = False,
    cost_fun: str = "sqrt",
    incremental_opt: bool = True,
    print_params: bool = True,
    cc: float = 0.3,
    new_layer_position: str = "random",
    new_layer_coef: float = 0.2,
    plot_cost_error: bool = False,
    num_seed=15,
    filename="prueba",
    save_seeds: bool = False,
    save_param_diff: bool = False,
    model="rotation",
    method_params: dict = {"n_iter": 800, "alpha": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
):

    num_layer = max_layers - min_layers + 1

    layer_list = list(range(min_layers, max_layers + 1))
    cost_array = np.zeros((num_seed, num_layer))
    l1_array = np.zeros((num_seed, num_layer))
    l2_array = np.zeros((num_seed, num_layer))
    fid_array = np.zeros((num_seed, num_layer))
    inf_array = np.zeros((num_seed, num_layer))
    mean_diff_array = np.zeros((num_seed, num_layer - 1))
    std_diff_array = np.zeros((num_seed, num_layer - 1))

    print("Comienzan los cálculos.")

    # Computación paralela
    MPI.pickle.__init__(dill.dumps, dill.loads)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    num_seed_node = num_seed // size

    seed_array = np.random.choice(
        range(rank * 2000, (rank + 1) * 2000), num_seed_node, replace=False
    )

    # Computación multiprocesador
    with ProcessingPool(cpu_count()) as p:
        # l2_list, l1_list, inf_list, fid_list, cost_list, mean_diff, std_diff
        results = p.map(
            lambda seed: graficas_errores(
                seed,
                min_layers=min_layers,
                max_layers=max_layers,
                x=x,
                f=f,
                grid_size=grid_size,
                function=function,
                f_params=f_params,
                interval=interval,
                int_method=int_method,
                opt_method=opt_method,
                φ_init=φ_init,
                show_plot=show_plot,
                show_final_plot=show_final_plot,
                show_error_plot=show_error_plot,
                show_diff=show_diff,
                print_cost=print_cost,
                cost_fun=cost_fun,
                incremental_opt=incremental_opt,
                print_params=print_params,
                cc=cc,
                new_layer_position=new_layer_position,
                new_layer_coef=new_layer_coef,
                plot_cost_error=plot_cost_error,
                method_params=method_params,
                probability=probability,
                model=model,
            ),
            seed_array,
        )

    print("Cálculos terminados.")

    l2_array = np.array([res[0] for res in results])  # array de dim (seeds_node)x(layers)
    l1_array = np.array([res[1] for res in results])
    inf_array = np.array([res[2] for res in results])
    fid_array = np.array([res[3] for res in results])
    cost_array = np.array([res[4] for res in results])
    mean_diff_array = np.array([res[5] for res in results])
    std_diff_array = np.array([res[6] for res in results])

    # dim sendbuf (results_node)x(seeds_node)x(layers)
    errors_send = np.array([l2_array, l1_array, inf_array, fid_array, cost_array])
    diff_send = np.array([mean_diff_array, std_diff_array])
    seeds_send = np.array([res[7] for res in results])

    errors_receive = None
    diff_receive = None
    seeds_receive = None

    if rank == 0:
        # aquí recibiremos los datos del resto de nodos
        errors_receive = np.zeros([size, 5, num_seed_node, num_layer])
        diff_receive = np.zeros([size, 2, num_seed_node, num_layer - 1])
        seeds_receive = np.zeros([size, num_seed_node])

    # comm.Barrier()   # wait for everybody to synchronize _here_
    comm.Gather(
        seeds_send, seeds_receive, root=0
    )  # El nodo 0 recibe el resto de arrays en una lista [array1, array2, ...]
    # comm.Barrier()   # wait for everybody to synchronize _here_
    comm.Gather(
        errors_send, errors_receive, root=0
    )  # El nodo 0 recibe el resto de arrays en una lista [array1, array2, ...]
    # comm.Barrier()
    comm.Gather(
        diff_send, diff_receive, root=0
    )  # El nodo 0 recibe el resto de arrays en una lista [array1, array2, ...]
    # comm.Barrier()

    if rank == 0:
        # dim results (nodes)x(results_node)x(seeds_node)x(layers)
        errors_receive = np.swapaxes(errors_receive, 1, 2)
        diff_receive = np.swapaxes(diff_receive, 1, 2)
        # dim results (seeds)x(results_node)x(layers)
        errors_results = np.concatenate(errors_receive, axis=0)
        diff_results = np.concatenate(diff_receive, axis=0)
        # dim results (results_node)x(seeds)x(layers)
        errors_results = np.swapaxes(errors_results, 0, 1)
        diff_results = np.swapaxes(diff_results, 0, 1)

        with open(filename + ".pkl", "wb") as file:
            pickle.dump(
                (
                    layer_list,
                    errors_results[0,],
                    errors_results[1,],
                    errors_results[2,],
                    errors_results[3,],
                    errors_results[4,],
                ),
                file,
            )

        if save_param_diff:
            with open(filename + "_param_diff.pkl", "wb") as file:
                pickle.dump((diff_results[0,], diff_results[1,]), file)

        seeds_results = np.concatenate(seeds_receive, axis=0)
        if save_seeds:
            with open(filename + "_seeds.pkl", "wb") as file:
                pickle.dump(seeds_results, file)

        """l2 = l2/num_seed
        l1 = l1/num_seed
        inf = inf/num_seed
        fid = fid/num_seed
        cost = cost/num_seed

        plot_errores(layer_list, l2, l1, inf, fid, cost, function)
        return layer_list, l2, l1, inf, fid, cost, seed_array"""
