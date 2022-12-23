# coding=UTF-8

import numpy as np
from numpy import pi
import matplotlib.style as style
import matplotlib.pyplot as plt

class Test_Functions(object):
    '''Usamos mayúslas para denotar clases y minúsculas para atributos.'''
    def __init__(self, function_type: str = 'gaussian'):
        style.use(['seaborn-whitegrid'])
        self.function = getattr(self, function_type)
        if function == 'gaussian':
            x = np.linspace(params['mean']-5*params['std'], params['mean']+5*params['std'], grid_size)
        else:
            x = np.linspace(interval[0], interval[1], grid_size) 

def gaussian(x: np.ndarray , mean: float = 0.0, std: float = 1, coef = None):
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

    """
    if coef is None:
        coef = (1/(std*np.sqrt(2*pi)))
    return coef*np.exp(-(x-mean)**2/(2*std**2))

def lorentzian(x: np.ndarray, x0: float = 0.0, γ: float = 1.0):
    return 1/pi*γ/((x-x0)**2+γ**2)

def sine(x: np.ndarray, a: float = 1.0, b: float = 0.0):
    return np.sin(a*x+b)

def step(x: np.ndarray, b: float = 0.0, coef: float = 1.0):
    return coef*np.heaviside(x, b)

def relu(x: np.ndarray, a: float = 1.0):
    if a<=0: raise ValueError('a must be a positive constant')
    return np.maximum(0,a*x)

def tanh(x: np.ndarray, a: float = 5.0, coef = 1.0):
    return coef*np.tanh(a*x)

def poly(x: np.ndarray):
    return np.abs((1-x**4)*3*x**3)

def cos2_sin2(x: np.ndarray, a: float = 1.0, b: float = 0.0):
    return np.cos(a*x+b)**2-np.sin(a*x+b)**2

def plot(x,f):
    plt.close('all')
    plt.fill_between(x,f, alpha=0.4)
    plt.plot(x,f)
    plt.title(function)
    plt.show()