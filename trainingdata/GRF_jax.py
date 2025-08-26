import jax
# import jax.numpy as np
import functools
import numpy as np
import numba

@numba.jit
def generate_GRF_parameters(N_samples, l, c_min, c_max, b_min, b_max):
    """
    Generate parameters for a Gaussian Random Field (GRF).
    Args:
        N_samples: Number of samples to generate.
        l: Length scale of the GRF (list or array).
        c_min: Minimum value for the scaling coefficients.
        c_max: Maximum value for the scaling coefficients.
        b_min: Minimum value for the translation coefficients.
        b_max: Maximum value for the translation coefficients.
    Returns:
        A tuple containing:
            - f_n: Coefficients of the GRF. Shape: (N_samples, n_mus).
            - mus: Locations of the GRF samples. Shape: (N_samples, n_mus, d).
            - l: Length scale of the GRF. Shape: (d,).
            - c: Scaling coefficients. Shape: (N_samples,).
            - b: Translation coefficients. Shape: (N_samples,).
    """
    l = np.array(l)  # Length scale of GRF
    d = len(l)  # Dimensionality of GRF
    n_mus = np.array(np.ceil(1 / np.prod(l)), int)  # Number of GRF sample locations
    mus = np.random.uniform(size=(N_samples, n_mus, d))

    # key = jax.random.PRNGKey(0)
    # keys = jax.random.split(key, N_samples)
    # mus = np.stack([jax.random.uniform(k, shape=(n_mus, d)) for k in keys])

    scaled_mus = mus / l[None, None, :]  # Scale mus by length scale
    diff = scaled_mus[:, None, :, :] - scaled_mus[:, :, None, :]  # Compute pairwise differences
    dist_sq = np.sum(diff**2, axis=-1)  # Compute squared distances
    cov = np.exp(-0.5 * dist_sq)  # Compute Gaussian covariance matrix
    L = np.linalg.cholesky(cov)  # Cholesky decomposition
    z = np.random.normal(size=(N_samples, n_mus))  # Generate random samples
    f = np.einsum('Nij,Nj->Ni', L, z, optimize=True)  # Sample from the GRF
    L_inv = np.linalg.inv(L)
    cov_inv = np.einsum('Nij,Njk->Nik', L_inv.transpose(0, 2, 1), L_inv, optimize=True)  # Inverse covariance
    f_n = np.einsum('Nij,Nj->Ni', cov_inv, f, optimize=True)  # Compute coefficients
    c = np.random.uniform(c_min, c_max, size=N_samples)  # Random scaling coefficients
    b = np.random.uniform(b_min, b_max, size=N_samples)  # Random translation coefficients
    l = l[None,:] # Shape (1, d) for broadcasting

    # GRFparams = {
    # "f_n": f_n,
    # "mus": mus,
    # "l": l,
    # "c": c,
    # "b": b
    # }
    return f_n, mus, l, c, b

@numba.jit
def GRFs_forward(GRFparams, x):
    """
    Evaluate Gaussian Random Fields on a shared set of query points.
    
    Args:
        GRFparams: Tuple (f_n, mus, l, c, b)
            - f_n: (N_samples, n_mus)
            - mus: (N_samples, n_mus, d)
            - l: (N_samples,d,)
            - c: (N_samples,)
            - b: (N_samples,)
        x: (N_points, d)
    
    Returns:
        GRF evaluations: (N_samples, N_points)
    """
    f_n, mus, l, c, b = GRFparams
    N_samples, n_mus, d = mus.shape
    N_points = x.shape[0]
    # Compute differences: (N_samples, N_points, n_mus, d)
    diff = (x[None, :, None, :] - mus[:, None, :, :]) / l[:, None, None, :]
    # Basis functions: (N_samples, N_points, n_mus)
    phi_n = np.exp(-0.5 * np.sum(diff**2, axis=-1))
    # Weighted sum with coefficients: (N_samples, N_points)
    f = np.einsum("Nn,Nqn->Nq", f_n, phi_n, optimize=True)
    # Scale and shift
    return c[:, None] * f + b[:, None]

@numba.jit
def GRFs_grad(GRFparams, x):
    """
    Evaluate gradient of Gaussian Random Fields at shared query points.
    
    Args:
        GRFparams: Tuple (f_n, mus, l, c, b)
            - f_n: (N_samples, n_mus)
            - mus: (N_samples, n_mus, d)
            - l: (d,)
            - c: (N_samples,)
            - b: (N_samples,)
        x: (N_points, d)
    
    Returns:
        Gradients: (N_samples, N_points, d)
    """
    f_n, mus, l, c, b = GRFparams
    # Differences: (N_samples, N_points, n_mus, d)
    diff = (x[None, :, None, :] - mus[:, None, :, :]) / l[:, None, None, :]
    # Basis functions: (N_samples, N_points, n_mus)
    phi = np.exp(-0.5 * np.sum(diff**2, axis=-1))
    # Prefactor for derivative: (N_samples, N_points, n_mus, d)
    prefactor = -(diff / (l[:, None, None, :]))
    # Gradient contributions: (N_samples, N_points, n_mus, d)
    grad_terms = prefactor * phi[:, :, :, None]
    # Weighted sum: (N_samples, N_points, d)
    grad = np.einsum("sn,sqnd->sqd", f_n, grad_terms, optimize=True)
    return c[:, None, None] * grad

@numba.jit
def GRFs_d2dxi2(GRFparams, x):
    """
    Evaluate the second derivative of the Gaussian Random Field w.r.t. each dimension.
    
    Args:
        GRFparams: Tuple (f_n, mus, l, c, b)
            - f_n: (N_samples, n_mus)
            - mus: (N_samples, n_mus, d)
            - l: (d,)
            - c: (N_samples,)
            - b: (N_samples,)
        x: (N_points, d)
    
    Returns:
        Second derivatives: (N_samples, N_points, d)
    """
    f_n, mus, l, c, b = GRFparams
    # Differences: (N_samples, N_points, n_mus, d)
    diff = (x[None, :, None, :] - mus[:, None, :, :]) / l[:, None, None, :]
    # Basis functions: (N_samples, N_points, n_mus)
    phi = np.exp(-0.5 * np.sum(diff**2, axis=-1))
    # Second derivative prefactor per dimension: (N_samples, N_points, n_mus, d)
    prefactor = (diff**2 / (l[:, None, None, :]**2) - 1) / (l[:, None, None, :]**2)
    # Multiply by basis
    d2_terms = prefactor * phi[:, :, :, None]
    # Sum over mus (keep dimension)
    d2_output = np.einsum("sn,sqnd->sqd", f_n, d2_terms, optimize=True)
    # Scale by c
    return c[:, None, None] * d2_output