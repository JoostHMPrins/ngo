import jax
# import jax.numpy as np
import functools
import numpy as np

import sys
sys.path.insert(0,'/home/prins/st8/prins/phd/git/ngo-pde-gk/trainingdata')
from GRF_jax import *
import numba

@numba.jit
def u_forward(GRFparams_u, x):
    """
    Evaluate the solution u at given points x.
    Args:
        GRFparams_u: Tuple containing (f_n, mus, l, c, b) where:
            - f_n: Coefficients of the GRF. Shape: (N_samples, n_mus).
            - mus: Locations of the GRF samples. Shape: (N_samples, n_mus, d).
            - l: Length scale of the GRF. Shape: (d,).
            - c: Scaling coefficients. Shape: (N_samples,).
            - b: Translation coefficients. Shape: (N_samples,).
        x: Points at which to evaluate the solution u. Shape: (N_samples, d).
    Returns:
        The evaluated solution u at points x. Shape: (N_samples,).
    """
    return GRFs_forward(GRFparams_u, x)

@numba.jit
def theta_forward(GRFparams_theta, x):
    """
    Evaluate the theta function at given points x.
    Args:
        GRFparams_theta: Tuple containing (f_n, mus, l, c, b) where:
            - f_n: Coefficients of the GRF. Shape: (N_samples, n_mus).
            - mus: Locations of the GRF samples. Shape: (N_samples, n_mus, d).
            - l: Length scale of the GRF. Shape: (d,).
            - c: Scaling coefficients. Shape: (N_samples,).
            - b: Translation coefficients. Shape: (N_samples,).
        x: Points at which to evaluate the theta function. Shape: (N_samples, d).
    Returns:
        The evaluated theta function at points x. Shape: (N_samples,).
    """
    return GRFs_forward(GRFparams_theta, x)


def f_forward(GRFparams_theta, GRFparams_u, x):
    """
    Evaluate the forcing term f at given points x.
    Args:
        GRFparams_theta: Tuple containing (f_n, mus, l, c, b) for theta.
        GRFparams_u: Tuple containing (f_n, mus, l, c, b) for u.
        x: Points at which to evaluate the forcing term f. Shape: (N_samples, d).
    Returns:
        The evaluated forcing term f at points x. Shape: (N_samples,).
    """
    return GRFs_grad(GRFparams_u, x)[:,:,0] - GRFs_forward(GRFparams_theta, x)*np.sum(GRFs_d2dxi2(GRFparams_u, x)[:,:,1:], axis=-1) - np.sum(GRFs_grad(GRFparams_theta, x)[:,:,1:]*GRFs_grad(GRFparams_u, x)[:,:,1:], axis=-1)


def eta_forward(GRFparams_theta, GRFparams_u, n, x):
    """
    Evaluate the eta function at given points x.
    Args:
        GRFparams_theta: Tuple containing (f_n, mus, l, c, b) for theta.
        GRFparams_u: Tuple containing (f_n, mus, l, c, b) for u.
        x: Points at which to evaluate the eta function. Shape: (N_samples, d).
    Returns:
        The evaluated eta function at points x. Shape: (N_samples,).
    """
    return np.einsum('i,Nq,Nqi->Nq', n, GRFs_forward(GRFparams_theta, x), GRFs_grad(GRFparams_u, x)[:,:,1:], optimize=True)


def g_forward(GRFparams_theta, GRFparams_u, x):
    """
    Evaluate the g function at given points x.
    Args:
        GRFparams_theta: Tuple containing (f_n, mus, l, c, b) for theta.
        GRFparams_u: Tuple containing (f_n, mus, l, c, b) for u.
        x: Points at which to evaluate the g function. Shape: (N_samples, d).
    Returns:
        The evaluated g function at points x. Shape: (N_samples,).
    """
    return GRFs_forward(GRFparams_theta, x)*GRFs_forward(GRFparams_u, x)

@numba.jit
def generate_manufactured_solutions(N_samples, l_min, l_max):
    """
    Generate manufactured solutions for the PDE problem.
    Args:
        N_samples: Number of samples to generate.
        l_min: Minimum length scale for the GRF.
        l_max: Maximum length scale for the GRF.
    Returns:
        A tuple containing:
            - GRFparams_u: Parameters for the solution u.
            - GRFparams_theta: Parameters for the theta function.
    """

    if l_min==l_max:    

        # Generate parameters for theta
        GRFparams_theta = generate_GRF_parameters(N_samples=N_samples, l=l_min, c_min=0, c_max=0.2, b_min=1, b_max=1)
        # Generate parameters for u
        GRFparams_u = generate_GRF_parameters(N_samples=N_samples, l=l_min, c_min=0, c_max=1, b_min=-1, b_max=1)

    if l_min!=l_max:
        n_mus = int(np.ceil(1/np.prod(l_min)))

        f_n_theta = np.zeros((N_samples, n_mus))
        mus_theta = np.zeros((N_samples, n_mus, len(l_min)))
        l_theta = np.random.uniform(l_min, l_max, size=(N_samples, len(l_min)))
        n_mus_theta_i = np.array(np.ceil(1/np.prod(l_theta, axis=-1)), dtype=int)
        c_theta = np.zeros((N_samples))
        b_theta = np.zeros((N_samples))

        f_n_u = np.zeros((N_samples, n_mus))
        mus_u = np.zeros((N_samples, n_mus, len(l_min)))
        l_u = np.random.uniform(l_min, l_max, size=(N_samples, len(l_min)))
        n_mus_u_i = np.array(np.ceil(1/np.prod(l_u, axis=-1)), dtype=int)
        c_u = np.zeros((N_samples))
        b_u = np.zeros((N_samples))

        for i in range(N_samples):
            f_n_theta[i,:n_mus_theta_i[i]], mus_theta[i,:n_mus_theta_i[i]], l_theta[i], c_theta[i], b_theta[i] = generate_GRF_parameters(1, l_theta[i], c_min=0, c_max=0.2, b_min=1, b_max=1)
            f_n_u[i,:n_mus_u_i[i]], mus_u[i,:n_mus_u_i[i]], l_u[i], c_u[i], b_u[i] = generate_GRF_parameters(1, l_u[i], c_min=0, c_max=1, b_min=-1, b_max=1)

        GRFparams_theta = (f_n_theta, mus_theta, l_theta, c_theta, b_theta)
        GRFparams_u = (f_n_u, mus_u, l_u, c_u, b_u)

        # GRFparams_theta = {
        # "f_n": f_n_theta,
        # "mus": mus_theta,
        # "l": l_theta,
        # "c": c_theta,
        # "b": b_theta
        # }

        # GRFparams_u = {
        # "f_n": f_n_u,
        # "mus": mus_u,
        # "l": l_u,
        # "c": c_u,
        # "b": b_u
        # }

    return GRFparams_u, GRFparams_theta


