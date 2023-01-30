# coding=UTF-8
from typing import Optional
import pickle

import numpy as np
from numpy import pi
import matplotlib.style as style
import matplotlib.pyplot as plt
from scipy.optimize import minimize, check_grad
import scipy.integrate as integrate

# from qiskit.algorithms.optimizers import SPSA
# from pathos.multiprocessing import ProcessingPool, cpu_count
# from mpi4py import MPI
# import dill
# from multiprocessing import Pool, cpu_count
def layer(x: np.ndarray, θi: np.ndarray, wi: float) -> tuple:
    """
    Single qubit rotations.

    Parmeters
    ---------
    x : (G) array
        Grid of independent variables used to approximate the function.
    θ : (L,3) array
        Bias parameters of every layer.
    w : (L) array
        Weights of every layer.

    Returns
    -------
    A : (L,G,2,2) array
    Unitary matrices of each layer.

    References
    ----------
    [1] Adrián Pérez Salinas et al, "Data re-uploading for a universal
        quantum classifier Quantum" 4, 226 (2020)

    """
    if type(x) is float:
        G = 1
    else:
        G = x.size

    ϕ1 = wi * x + θi[0] * np.ones(G)
    ϕ2 = θi[1] * np.ones(G)
    ϕ3 = θi[2] * np.ones(G)

    Ui = np.asarray(
        [
            [
                np.cos(ϕ1 / 2) * np.exp(1j * (ϕ2 + ϕ3) / 2),
                -np.sin(ϕ1 / 2) * np.exp(-1j * (ϕ2 - ϕ3) / 2),
            ],
            [
                np.sin(ϕ1 / 2) * np.exp(1j * (ϕ2 - ϕ3) / 2),
                np.cos(ϕ1 / 2) * np.exp(-1j * (ϕ2 + ϕ3) / 2),
            ],
        ]
    )
    return np.moveaxis(
        Ui, -1, 0
    )  # move last axis to the first position keeping the order of the rest axis


def layer_rot(x: np.ndarray, θi: np.ndarray, wi: float) -> tuple:
    if type(x) is float:
        G = 1
    else:
        G = x.size

    ϕ1 = wi * x + θi[0] * np.ones(G)
    ϕ2 = θi[1]
    ϕ3 = θi[2]

    Rx = np.asarray(
        [[np.cos(ϕ1 / 2), -1j * np.sin(ϕ1 / 2)], [-1j * np.sin(ϕ1 / 2), np.cos(ϕ1 / 2)]]
    )
    Ry = np.array([[np.cos(ϕ2 / 2), -np.sin(ϕ2 / 2)], [np.sin(ϕ2 / 2), np.cos(ϕ2 / 2)]])
    Rz = np.array(
        [
            [np.cos(ϕ3 / 2) - 1j * np.sin(ϕ3 / 2), 0],
            [0, np.cos(ϕ3 / 2) + 1j * np.sin(ϕ3 / 2)],
        ]
    )

    Ui = np.einsum("mn,np,pqi->mqi", Rz, Ry, Rx)

    # move last axis to the first position keeping the order of the rest axiss
    return np.moveaxis(Ui, -1, 0)


def evalua_modelo(
    x: np.ndarray, θ: np.ndarray, w: np.ndarray, probability=True, model="rotation"
):
    """
    Given an array of unitaries, compute the function f(x) that
    it represents.

    Parameters
    ----------

    x : (L) array
        Points in the grid
    θ : (M, 3) array
        Variational parameters
    w : (M) array
        variational parameters
    probability : True if f(x) >= 0

    Returns
    -------
    fi : estimate of f(xi) on all elements of 'x'

    """
    L = w.size
    if model == "rotation":
        U = layer_rot(x, θ[:, 0], w[0])[:, :, 0]
        for i in range(1, L):
            Ui = layer_rot(x, θ[:, i], w[i])
            U = np.einsum("imn,in->im", Ui, U)
    else:
        U = layer(x, θ[:, 0], w[0])[:, :, 0]
        for i in range(1, L):
            Ui = layer(x, θ[:, i], w[i])
            U = np.einsum("imn,in->im", Ui, U)

    return (U[:, 0] * np.conjugate(U[:, 0])).real if probability else U[:, 0]


def coste(x, f, θ, w, probability=True, model="rotation"):
    """Cost function of the model.

    Parameters
    ----------
    x : (G) array
        Points in the grid
    f : (G) array
        Points of the function evaluated in the grid
    θ : (M, 3) array
        Variational bias parameter.
    w : (M) array
        Variational weight parameters.
    probability : bool
        Function is a probability density.

    Returns
    -------
    (L) array with the mean square error for each x in the grid.

    """
    f_approx = evalua_modelo(x, θ, w, probability, model=model)
    return np.mean(np.abs(f_approx - f) ** 2)


def der_layer(x: np.ndarray, θi: np.ndarray, wi: float, model="rotation") -> tuple:
    if type(x) is float:
        G = 1
    else:
        G = x.size

    if model == "rotation":
        ϕ1 = wi * x + θi[0] * np.ones(G)
        ϕ2 = θi[1]
        ϕ3 = θi[2]

        Rx = np.asarray(
            [
                [np.cos(ϕ1 / 2), -1j * np.sin(ϕ1 / 2)],
                [-1j * np.sin(ϕ1 / 2), np.cos(ϕ1 / 2)],
            ]
        )
        Ry = np.array(
            [[np.cos(ϕ2 / 2), -np.sin(ϕ2 / 2)], [np.sin(ϕ2 / 2), np.cos(ϕ2 / 2)]]
        )
        Rz = np.array(
            [
                [np.cos(ϕ3 / 2) - 1j * np.sin(ϕ3 / 2), 0],
                [0, np.cos(ϕ3 / 2) + 1j * np.sin(ϕ3 / 2)],
            ]
        )

        DRx = (
            1
            / 2
            * np.asarray(
                [
                    [-np.sin(ϕ1 / 2), -1j * np.cos(ϕ1 / 2)],
                    [-1j * np.cos(ϕ1 / 2), -np.sin(ϕ1 / 2)],
                ]
            )
        )
        DRy = (
            1
            / 2
            * np.array(
                [[-np.sin(ϕ2 / 2), -np.cos(ϕ2 / 2)], [np.cos(ϕ2 / 2), -np.sin(ϕ2 / 2)]]
            )
        )
        DRz = (
            1
            / 2
            * np.array(
                [
                    [-1j * np.cos(ϕ3 / 2) - np.sin(ϕ3 / 2), 0],
                    [0, 1j * np.cos(ϕ3 / 2) - np.sin(ϕ3 / 2)],
                ]
            )
        )

        Dx = np.einsum("mn,np,pqi->imq", Rz, Ry, DRx)
        Dw = np.einsum("mn,np,pqi,i->imq", Rz, Ry, DRx, x)
        Dy = np.einsum("mn,np,pqi->imq", Rz, DRy, Rx)
        Dz = np.einsum("mn,np,pqi->imq", DRz, Ry, Rx)

    else:
        Dx = 1 / 2 * layer(x, np.array([θi[0] + np.pi, θi[1], θi[2]]), wi)
        Dw = np.einsum("imn,i->imn", Dx, x)
        Dy = 1 / 2 * layer(x, np.array([θi[0], θi[1] + np.pi, θi[2]]), wi)
        Dz = 1 / 2 * layer(x, np.array([θi[0], θi[1], θi[2] + np.pi]), wi)

    return np.array([Dw, Dx, Dy, Dz])


def der_net(x: np.ndarray, θ: np.ndarray, w: np.ndarray, model="rotation"):
    """ "Create recursively the derivatives with respect to each parameter of the entire net."""

    L = w.size
    G = x.size
    A = np.tensordot(np.ones(G), np.identity(2), axes=0)  # dim (G,2,2)
    D = np.zeros((L, 4, G, 2, 2), dtype=np.complex128)

    for i in range(L):
        DUi = der_layer(x, θ[:, i], w[i], model=model)  # dim (4,G,2,2)
        D[i, ...] = np.einsum(
            "jimn,inp->jimp", DUi, A
        )  # j es cada una de las derivadas
        # Multiply derivative times next layer
        if model == "rotation":
            Ui = layer_rot(x, θ[:, i], w[i])
        else:
            Ui = layer(x, θ[:, i], w[i])
        A = np.einsum("imn,inp->imp", Ui, A)

    # En la primera iteración reaprovechamos el Ui de la capa L
    B = Ui

    for i in range(L - 2, -1, -1):
        D[i, ...] = np.einsum("imn,jinp->jimp", B, D[i, ...])
        # Multiply derivative times previous layer
        if model == "rotation":
            Ui = layer_rot(x, θ[:, i], w[i])
        else:
            Ui = layer(x, θ[:, i], w[i])
        B = np.einsum("imn,inp->imp", B, Ui)
    # D is shape (L,4,G,2,2). We also return the model
    return D, A


def der_coste(x, f, θ, w, probability=True, return_cost=False, model="rotation"):
    """ "Returns the gradient of the cost function with respect to each parameter."""

    L = w.size
    G = x.size

    if probability:
        ders, A = der_net(x, θ, w, model=model)
        U = A[:, 0, 0]
        E = U * np.conj(U) - f
        # índice i layers, j parametro
        der_C = (
            4
            / G
            * np.array(
                [
                    [
                        np.dot(E.real, np.real(np.conj(U) * ders[i, j, :, 0, 0]))
                        for i in range(L)
                    ]
                    for j in range(4)
                ]
            )
        )

    else:
        ders, A = der_net(x, θ, w, model=model)
        U = A[:, 0, 0]
        E = U - f  # error in approximation
        der_C = (
            2
            / G
            * np.array(
                [
                    [np.real(np.dot(np.conj(E), ders[i, j, :, 0, 0])) for i in range(L)]
                    for j in range(4)
                ]
            )
        )

    # devolvemos un array con la misma estructura que ϕ = [w, θ_0, θ_1, θ_2]
    if return_cost:
        if probability:
            return der_C.flatten(), np.mean((np.abs(U * np.conjugate(U)).real - f) ** 2)
        else:
            return der_C.flatten(), np.mean(np.abs(U - f) ** 2)
    else:
        return der_C.flatten()


def coste_sqrt(x, f, θ, w, probability, model="rotation"):
    """Cost function of the model.

    Parameters
    ----------
    x : (G) array
        Points in the grid
    f : (G) array
        Points of the function evaluated in the grid
    θ : (M, 3) array
        Variational bias parameter.
    w : (M) array
        Variational weight parameters.
    probability : bool
        Function is a probability density.

    Returns
    -------
    (L) array with the mean square error for each x in the grid.

    """
    return np.sqrt(coste(x, f, θ, w, probability=probability, model=model))


def der_coste_sqrt(x, f, θ, w, probability=True, return_cost=False, model="rotation"):
    """ "Returns the gradient of the cost function with respect to each parameter."""

    L = w.size
    G = x.size

    if probability:
        ders, A = der_net(x, θ, w, model=model)
        U = A[:, 0, 0]
        E = (U * np.conj(U)).real - f
        # índice i layers, j parametro
        der_C = (
            2
            / (np.sqrt(G) * np.sqrt(np.sum(np.abs(E) ** 2) + 1e-9))
            * np.array(
                [
                    [
                        np.dot(E, np.real(np.conj(U) * ders[i, j, :, 0, 0]))
                        for i in range(L)
                    ]
                    for j in range(4)
                ]
            )
        )

    else:
        ders, A = der_net(x, θ, w, model=model)
        U = A[:, 0, 0]
        E = U - f  # error in approximation
        der_C = (
            1
            / (np.sqrt(G) * np.sqrt(np.sum(np.abs(E) ** 2) + 1e-9))
            * np.array(
                [
                    [np.real(np.dot(np.conj(E), ders[i, j, :, 0, 0])) for i in range(L)]
                    for j in range(4)
                ]
            )
        )

    # devolvemos un array con la misma estructura que ϕ = [w, θ_0, θ_1, θ_2]
    if return_cost:
        if probability:
            return der_C.flatten(), np.mean((np.abs(U * np.conjugate(U)).real - f) ** 2)
        else:
            return der_C.flatten(), np.mean(np.abs(U - f) ** 2)
    else:
        return der_C.flatten()


def split(φ):
    layers = φ.size // 4
    return φ[0:layers], φ[layers:].reshape(3, layers)


def blackbox_minimizer(
    x,
    f,
    φ_init,
    probability: bool,
    opt_method: str = "L-BFGS-B",
    print_cost: bool = False,
    cost_fun="sqrt",
    model="rotation",
):

    if cost_fun == "sqrt":
        cost_function = globals()["coste_sqrt"]
        der_cost_function = globals()["der_coste_sqrt"]
    elif cost_fun == "normal":
        cost_function = globals()["coste"]
        der_cost_function = globals()["der_coste"]

    def coste_intermedio(φ):
        w, θ = split(φ)
        c = cost_function(x, f, θ, w, probability, model=model)
        if print_cost:
            print("Valor f. coste: ", c)
        return c

    def der_coste_intermedio(φ):
        w, θ = split(φ)
        der_c = der_cost_function(x, f, θ, w, probability, model=model)
        if print_cost:
            print("Valor der. coste: ", der_c)
        return der_c

    return minimize(
        coste_intermedio,
        φ_init,
        method=opt_method,
        jac=der_coste_intermedio,
        tol=1e-12,
        options={"maxiter": 10000},
    )


def adam_minimizer(
    x,
    f,
    φ,
    probability: bool,
    print_cost: bool = True,
    plot_cost=False,
    cost_fun="sqrt",
    model="rotation",
    n_iter=800,
    alpha=0.01,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
):
    """
        Parameters
        ----------
        n_iter : int
        Number of iterations of the optimization algorithm
    alpha : float
        steps size
    beta1 : float
        factor for average gradient
        beta2 : float
                factor for average squared gradient

    """
    if cost_fun == "sqrt":
        der_cost_function = globals()["der_coste_sqrt"]
    elif cost_fun == "normal":
        der_cost_function = globals()["der_coste"]

    num_params = φ.size
    # initialize first and second moments
    m = np.zeros(num_params)
    v = np.zeros(num_params)
    # Model parameters
    min_cost = 10
    cost = np.zeros(n_iter)
    for t in range(n_iter):
        w, θ = split(φ)
        # g, cost = der_coste(x, f, θ, w, probability, return_cost = True)
        g, cost[t] = der_cost_function(
            x, f, θ, w, probability, return_cost=True, model=model
        )
        if cost[t] < min_cost:
            min_cost = cost[t]
            min_t = t
            min_φ = φ
        if print_cost:
            print("φ = {φ}  ,  cost = {cost}".format(φ=φ, cost=cost[t]))
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g**2
        mhat = m / (1.0 - beta1 ** (t + 1))
        vhat = v / (1.0 - beta2 ** (t + 1))
        φ = φ - alpha * mhat / (np.sqrt(vhat) + eps)
    # print('El coste mínimo alcanzado es {min_cost} en la iteración {min_t}.'.format(min_cost=min_cost, min_t=min_t))
    # Devolvemos el φ que minimiza la función de coste
    if plot_cost:
        plt.plot(range(n_iter), cost)
        plt.yscale("log")
        plt.show()
    return min_φ


def train_perceptron(
    x: np.ndarray,
    f: np.ndarray,
    layers: int = 4,
    opt_method: str = "L-BFGS-B",
    method_params: dict = {
        "n_iter": 800,
        "alpha": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
    },
    seed: float = 2.0,
    φ_init: Optional[np.ndarray] = None,
    print_cost: bool = False,
    show_plot=True,
    cc=0.3,
    probability: Optional[np.ndarray] = None,
    plot_cost: bool = False,
    cost_fun: str = "sqrt",
    plot_title: str = "",
    model="rotation",
):

    if φ_init is None:
        np.random.seed(seed)
        φ_init = cc * np.random.randn(layers + 3 * layers)

    if probability is None:
        probability = (f >= 0).all()

    if opt_method == "ADAM":
        φ = adam_minimizer(
            x,
            f,
            φ_init,
            probability=probability,
            print_cost=print_cost,
            plot_cost=plot_cost,
            cost_fun=cost_fun,
            model=model,
            **method_params
        )
        result = 0  # resultado a 0 por defecto en este método
    else:
        result = blackbox_minimizer(
            x,
            f,
            φ_init,
            opt_method=opt_method,
            probability=probability,
            print_cost=print_cost,
            cost_fun=cost_fun,
            model=model,
        )
        φ = result.x

    if show_plot:
        ω, θ = split(φ)
        f_approx = evalua_modelo(x, θ, ω, probability, model=model)
        plt.close("all")
        plt.plot(x, f)
        plt.plot(x, f_approx.real)
        plt.title(plot_title)
        plt.show()

    return φ, result


class Test_Functions:
    """Usamos mayúslas para denotar clases y minúsculas para atributos."""

    def __init__(self):
        style.use(["seaborn-whitegrid"])

    def gaussian(x: np.ndarray, mean: float = 0.0, std: float = 0.2, coef=1):
        """
        Approximate and plot a gaussian function.

        Parameters
        ----------
        x : np.ndarray
            Grid in which to approximate the function.
        mean : float
            Mean of the gaussian.
        std : float
            Standard deviation.
        coef : float
            Factor that multiplies the gaussian.

        Returns
        -------
        f : Function

        """
        if coef is None:
            coef = 1 / (std * np.sqrt(2 * pi))
        return coef * np.exp(-((x - mean) ** 2) / (2 * std**2))

    def lorentzian(x: np.ndarray, x0: float = 0.0, γ: float = 0.2, coef=None):
        if coef is None:
            coef = γ
        return coef * γ / ((x - x0) ** 2 + γ**2)

    def sine(x: np.ndarray, a: float = 1.0, b: float = 0.0):
        return np.sin(a * x + b)

    def cos(x: np.ndarray, a: float = 1.0, b: float = 0.0):
        return np.cos(a * x + b)

    def step(x: np.ndarray, b: float = 0.0, coef: float = 1.0):
        return coef * np.heaviside(x, b)

    def relu(x: np.ndarray, a: float = 1.0):
        if a <= 0:
            raise ValueError("a must be a positive constant")
        return np.maximum(0, a * x)

    def tanh(x: np.ndarray, a: float = 5.0, coef=1.0):
        return coef * np.tanh(a * x)

    def poly(x: np.ndarray):
        return -4 * x**2 * (x**2 - 1)

    def poly2(x: np.ndarray):
        return np.abs((1 - x**4) * 3 * x**3)

    def cos2_sin2(x: np.ndarray, a: float = 1.0, b: float = 0.0):
        return np.cos(a * x + b) ** 2 - np.sin(a * x + b) ** 2

    def plot(
        self,
        grid_size: int = 31,
        function: str = "gaussian",
        params: dict = {"mean": 0.0, "std": 0.5, "coef": None},
        interval: tuple = (-1, 1),
        show_plot=True,
    ):

        if function == "gaussian":
            x = np.linspace(
                params["mean"] - 5 * params["std"],
                params["mean"] + 5 * params["std"],
                grid_size,
            )
        else:
            x = np.linspace(interval[0], interval[1], grid_size)
        fun = getattr(Test_Functions, function)
        f = fun(x, **params)

        if show_plot:
            plt.close("all")
            plt.plot(x, f)
            plt.title(function)
            plt.show()
        return x, f


def error_perceptron(
    φ: np.ndarray,
    function: str = "gaussian",
    f_params: dict = {"mean": 0.0, "std": 0.5, "coef": None},
    interval: tuple = (-1, 1),
    method: str = "quad",
    probability: bool = True,
    model="rotation",
):
    """ "
    Error in the approximation of the function by the qubit perceptron.
    Returns the error measured in different norms.

    """
    layers = φ.size // 4
    w, θ = φ[0:layers], φ[layers:].reshape(3, layers)

    # Seleccionamos la función a aproximar
    fun = getattr(Test_Functions, function)
    # Norma L2
    diff_l2 = (
        lambda x: (
            np.abs(
                fun(x, **f_params) - evalua_modelo(x, θ, w, probability, model=model)
            )
        )
        ** 2
    )
    f2_theo = lambda x: fun(x, **f_params) ** 2
    # Norma L1
    diff_abs = lambda x: np.abs(
        fun(x, **f_params) - evalua_modelo(x, θ, w, probability, model=model)
    )
    f_theo_abs = lambda x: np.abs(fun(x, **f_params))
    # Fidelity (no hace falta conjugar f porque el modelo es real)
    prod_re = lambda x: np.real(
        fun(x, **f_params) * evalua_modelo(x, θ, w, probability, model=model)
    )
    prod_im = lambda x: np.imag(
        fun(x, **f_params) * evalua_modelo(x, θ, w, probability, model=model)
    )
    f2_approx = lambda x: np.abs(evalua_modelo(x, θ, w, probability, model=model)) ** 2
    # Norma infinito
    y = np.linspace(interval[0], interval[1], 10000)
    error_inf = np.max(
        np.abs(fun(y, **f_params) - evalua_modelo(y, θ, w, probability, model=model))
    )
    # Seleccionamos el método de integración
    if method == "simpson":
        # L2 calculation
        error_l2 = np.sqrt(
            integrate.simpson(diff_l2(y), y)
        )  # /np.sqrt(integrate.simpson(f2_theo(y), y))
        # L2 calculation
        error_l1 = integrate.simpson(
            diff_abs(y), y
        )  # /integrate.simpson(f_theo_abs(y), y)
        # Fidelity calculation
        int_prod_squared = (
            integrate.simpson(prod_re(y), y) ** 2
            + integrate.simpson(prod_im(y), y) ** 2
        )
        error_infid = 1 - int_prod_squared / (
            integrate.simpson(f2_approx(y), y) * integrate.simpson(f2_theo(y), y)
        )
        return error_l2, error_l1, error_inf, error_infid

    elif method == "quad":
        error_l2 = np.sqrt(
            integrate.quad(diff_l2, interval[0], interval[1], limit=300)[0]
        )
        error_l1 = integrate.quad(diff_abs, interval[0], interval[1], limit=300)[0]
        int_prod_squared = (
            integrate.quad(prod_re, interval[0], interval[1], limit=300)[0] ** 2
            + integrate.quad(prod_im, interval[0], interval[1], limit=300)[0] ** 2
        )
        error_infid = 1 - int_prod_squared / (
            integrate.quad(f2_approx, interval[0], interval[1], limit=300)[0]
            * integrate.quad(f2_theo, interval[0], interval[1], limit=300)[0]
        )
        return error_l2, error_l1, error_inf, error_infid
    else:
        raise ValueError("Solo está permitido usar quad o simpson para la integral.")


def plot_errores(
    layer_list, l2_list, l1_list, inf_list, infid_list, cost_error, function
):
    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        layer_list,
        l2_list,
        linestyle="-",
        marker="o",
        markersize=6,
        color="#1f77b4",
        label="L2 norm",
    )
    ax.plot(
        layer_list,
        l1_list,
        linestyle="-",
        marker="^",
        markersize=6,
        color="#ff7f0e",
        label="L1 norm",
    )
    ax.plot(
        layer_list,
        inf_list,
        linestyle="-",
        marker="D",
        markersize=6,
        color="#2ca02c",
        label="Infinity norm",
    )
    ax.plot(
        layer_list,
        infid_list,
        linestyle="-",
        marker="*",
        markersize=6,
        color="crimson",
        label="Infidelity",
    )
    ax.plot(
        layer_list,
        cost_error,
        linestyle="-",
        marker="*",
        markersize=6,
        color="olive",
        label="Coste",
    )

    ax.set_title("Error vs Number of layers (" + function + ")")
    ax.set_xlabel("Layers")
    ax.set_ylabel("Error")
    ax.legend(loc="upper right", fontsize="large")
    plt.yscale("log")
    plt.show()


def graficas_errores(
    seed,
    min_layers,
    max_layers,
    x: Optional[np.ndarray] = None,
    f: Optional[np.ndarray] = None,
    grid_size: int = 31,
    function: str = "gaussian",
    f_params: dict = {"mean": 0.0, "std": 2, "coef": 1},
    interval: tuple = (-1, 1),
    int_method: str = "quad",
    opt_method: str = "L-BFGS-B",
    φ_init: Optional[np.ndarray] = None,
    probability: Optional[bool] = None,
    show_plot: bool = False,
    show_final_plot: bool = True,
    show_error_plot: bool = True,
    show_diff=False,
    print_cost: bool = False,
    cost_fun: str = "sqrt",
    incremental_opt: bool = True,
    print_params: bool = False,
    cc: float = 0.3,
    new_layer_position: str = "random",
    new_layer_coef: float = 0.2,
    plot_cost_error: bool = False,
    model="rotation",
    method_params: dict = {
        "n_iter": 800,
        "alpha": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
    },
):

    Test = Test_Functions()
    l2_list, l1_list, inf_list, infid_list = [], [], [], []
    layer_list = list(range(min_layers, max_layers + 1))

    if x is None and f is None:
        x, f = Test.plot(
            grid_size, function, f_params, interval=interval, show_plot=show_plot
        )
    if probability is None:
        probability = (f >= 0).all()

    np.random.seed(seed)
    if φ_init is None:
        # cc deberá ser un valor pequeño en principio
        φ = cc * np.random.randn(min_layers + 3 * min_layers)
        if print_params:
            print("Parámetros iniciales: ", φ)
    else:
        φ = φ_init

    mean_diff = np.zeros(max_layers - min_layers)
    std_diff = np.zeros(max_layers - min_layers)
    cost_error = np.zeros(max_layers - min_layers + 1)

    for i, layer in enumerate(layer_list):
        φ, result = train_perceptron(
            x,
            f,
            layers=layer,
            probability=probability,
            opt_method=opt_method,
            seed=seed,
            φ_init=φ,
            show_plot=show_plot,
            method_params=method_params,
            print_cost=print_cost,
            plot_title=function + " optimized with " + opt_method,
            cost_fun=cost_fun,
            model=model,
        )
        # print('Los parámetros óptimos en la capa {layer} son {φ}.\n'.format(layer = layer, φ=φ))
        error_l2, error_l1, error_inf, error_infid = error_perceptron(
            φ, function, f_params, interval, int_method, probability, model=model
        )

        ω, θ = split(φ)
        cost_error[i] = coste(x, f, θ, ω, probability, model=model)
        # Guardamos la diferencia entre el φ optimizado de la anterior capa y de esta
        if layer > min_layers:
            if new_layer_position == "final":
                diff_φ = (
                    φ.reshape(4, layer).T.flatten()[0 : 4 * (layer - 1)]
                    - φ_old.reshape(4, layer - 1).T.flatten()
                )
                mean_diff[i - 1] = np.mean(np.abs(diff_φ))
                std_diff[i - 1] = np.std(np.abs(diff_φ))
            elif new_layer_position == "initial":
                diff_φ = (
                    φ.reshape(4, layer).T.flatten()[4 : 4 * layer]
                    - φ_old.reshape(4, layer - 1).T.flatten()
                )
                mean_diff[i - 1] = np.mean(np.abs(diff_φ))
                std_diff[i - 1] = np.std(np.abs(diff_φ))
        φ_old = φ

        l2_list.append(error_l2)
        l1_list.append(error_l1)
        inf_list.append(error_inf)
        infid_list.append(error_infid)

        if layer == max_layers:
            if show_final_plot:
                ω, θ = φ[0:layer], φ[layer:].reshape(3, layer)
                f_approx = evalua_modelo(x, θ, ω, probability, model=model)
                plt.close("all")
                plt.plot(x, f)
                plt.plot(x, f_approx.real)
                plt.title(function + " optimization with " + opt_method)
                plt.yscale("log")
                plt.show()
            break
        if incremental_opt is True:
            # Inicializamos una nueva capa en la posición indicada
            if new_layer_position == "random":
                i = np.random.randint(0, high=layer + 1, dtype=int)
            elif new_layer_position == "final":
                i = layer
            elif new_layer_position == "initial":
                i = 0
            elif new_layer_position == "middle":
                i = min_layers + (layer - min_layers) // 2
            else:
                raise ValueError(
                    "El valor de new_layer_position = {a} no es válido.".format(
                        a=new_layer_position
                    )
                )
            # Añadimos la nueva capa con valores cercanos a 0
            new_layer_val = new_layer_coef * np.random.randn(4)
            # new_layer_val = 0.3/(i+1) * np.random.randn(4)
            φ = np.insert(
                φ, i, new_layer_val[0]
            )  # phi [w1, ...wn, theta1, theta2, theta3]
            φ = np.insert(φ, i + 1 + layer, new_layer_val[1])
            φ = np.insert(φ, i + 2 + 2 * layer, new_layer_val[2])
            φ = np.insert(φ, i + 3 + 3 * layer, new_layer_val[3])
        else:
            φ = cc * np.random.randn(layer + 1 + 3 * layer + 3)
        # print('Los parámetros con capa añadida son {φ}.\n'.format(φ=φ))

    if print_params:
        print("Parámetros finales: ", φ)
    # Hacemos una integración numérica para calcular el error de la aproximación
    if show_error_plot:
        plot_errores(
            layer_list, l2_list, l1_list, inf_list, infid_list, cost_error, function
        )

    if plot_cost_error:
        plt.close()
        plt.figure(figsize=(6, 5), dpi=80)
        plt.plot(layer_list, cost_error, ls="--", marker="^", ms=14)
        plt.yscale("log")
        plt.show()

    return l2_list, l1_list, inf_list, infid_list, cost_error, mean_diff, std_diff, seed


def mean_seed_errores(
    min_layers,
    max_layers,
    x: Optional[np.ndarray] = None,
    f: Optional[np.ndarray] = None,
    grid_size: int = 31,
    function: str = "gaussian",
    f_params: dict = {"mean": 0.0, "std": 2, "coef": 1},
    interval: tuple = (-1, 1),
    int_method: str = "quad",
    opt_method: str = "L-BFGS-B",
    φ_init: Optional[np.ndarray] = None,
    probability: Optional[bool] = None,
    show_plot: bool = False,
    show_final_plot: bool = True,
    show_error_plot: bool = True,
    show_diff=False,
    print_cost: bool = False,
    cost_fun: str = "sqrt",
    incremental_opt: bool = True,
    print_params: bool = False,
    cc: float = 0.3,
    new_layer_position: str = "random",
    new_layer_coef: float = 0.2,
    plot_cost_error: bool = False,
    num_seed=15,
    filename="prueba",
    model="rotation",
    method_params: dict = {
        "n_iter": 800,
        "alpha": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
    },
):

    num_layer = max_layers - min_layers + 1
    l2, l1, inf, fid, cost = (
        np.zeros(num_layer),
        np.zeros(num_layer),
        np.zeros(num_layer),
        np.zeros(num_layer),
        np.zeros(num_layer),
    )

    layer_list = list(range(min_layers, max_layers + 1))
    cost_array = np.zeros((num_seed, num_layer))
    l1_array = np.zeros((num_seed, num_layer))
    l2_array = np.zeros((num_seed, num_layer))
    fid_array = np.zeros((num_seed, num_layer))
    inf_array = np.zeros((num_seed, num_layer))
    mean_diff_array = np.zeros((num_seed, num_layer - 1))
    std_diff_array = np.zeros((num_seed, num_layer - 1))
    seed_array = np.random.choice(range(0, 2000), num_seed, replace=False)

    for i, seed in enumerate(seed_array):
        # layer_list, l2_list, l1_list, inf_list, fid_list, cost_list, mean_diff, std_diff = graficas_intermedio(seed)
        (
            l2_list,
            l1_list,
            inf_list,
            fid_list,
            cost_list,
            mean_diff,
            std_diff,
            seed,
        ) = graficas_errores(
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
        )
        # Seeds en el eje 0 y capas en el eje 1. Queremos las seeds en cada box plot.'''
        cost_array[i, :] = np.array(cost_list)
        l1_array[i, :] = np.array(l1_list)
        l2_array[i, :] = np.array(l2_list)
        fid_array[i, :] = np.array(fid_list)
        inf_array[i, :] = np.array(inf_list)
        mean_diff_array[i, :] = mean_diff
        std_diff_array[i, :] = std_diff

    with open(filename + ".pkl", "wb") as file:
        pickle.dump(
            (
                layer_list,
                l2_array,
                l1_array,
                inf_array,
                fid_array,
                cost_array,
                seed_array,
            ),
            file,
        )

    with open(filename + "_param_diff.pkl", "wb") as file:
        pickle.dump((mean_diff_array, std_diff_array), file)

    l2 = l2 / num_seed
    l1 = l1 / num_seed
    inf = inf / num_seed
    fid = fid / num_seed
    cost = cost / num_seed

    plot_errores(layer_list, l2, l1, inf, fid, cost, function)
    return layer_list, l2, l1, inf, fid, cost, seed_array
