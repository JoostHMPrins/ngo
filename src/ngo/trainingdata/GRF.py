# import functools
# import numpy as np
# import scipy


# def generate_GRF_parameters(N_samples, l, c_min, c_max, b_min, b_max):
#     """
#     Generate parameters for a Gaussian Random Field (GRF).
#     Args:
#         N_samples: Number of samples to generate.
#         l: Length scale of the GRF (list or array).
#         c_min: Minimum value for the scaling coefficients.
#         c_max: Maximum value for the scaling coefficients.
#         b_min: Minimum value for the translation coefficients.
#         b_max: Maximum value for the translation coefficients.
#     Returns:
#         A tuple containing:
#             - f_n: Coefficients of the GRF. Shape: (N_samples, n_mus).
#             - mus: Locations of the GRF samples. Shape: (N_samples, n_mus, d).
#             - l: Length scale of the GRF. Shape: (d,).
#             - c: Scaling coefficients. Shape: (N_samples,).
#             - b: Translation coefficients. Shape: (N_samples,).
#     """
#     l = np.array(l)  # Length scale of GRF
#     d = len(l)  # Dimensionality of GRF
#     n_mus = np.array(np.ceil(1 / np.prod(l)), int)  # Number of GRF sample locations
#     mus = np.random.uniform(size=(N_samples, n_mus, d))

#     # key = jax.random.PRNGKey(0)
#     # keys = jax.random.split(key, N_samples)
#     # mus = np.stack([jax.random.uniform(k, shape=(n_mus, d)) for k in keys])

#     scaled_mus = mus / l[None, None, :]  # Scale mus by length scale
#     diff = scaled_mus[:, None, :, :] - scaled_mus[:, :, None, :]  # Compute pairwise differences
#     dist_sq = np.sum(diff**2, axis=-1)  # Compute squared distances
#     cov = np.exp(-0.5 * dist_sq)  # Compute Gaussian covariance matrix
#     L = np.linalg.cholesky(cov)  # Cholesky decomposition
#     z = np.random.normal(size=(N_samples, n_mus))  # Generate random samples
#     f = np.einsum('Nij,Nj->Ni', L, z, optimize=True)  # Sample from the GRF
#     L_inv = np.linalg.inv(L)
#     cov_inv = np.einsum('Nij,Njk->Nik', L_inv.transpose(0, 2, 1), L_inv, optimize=True)  # Inverse covariance
#     f_n = np.einsum('Nij,Nj->Ni', cov_inv, f, optimize=True)  # Compute coefficients
#     c = np.random.uniform(c_min, c_max, size=N_samples)  # Random scaling coefficients
#     b = np.random.uniform(b_min, b_max, size=N_samples)  # Random translation coefficients
#     l = np.tile(l,(N_samples, 1)) # Shape (N_samples, d) for broadcasting
#     GRFparams = f_n, mus, l, c, b
#     return GRFparams

# def GRFs_forward(GRFparams, x):
#     """
#     Evaluate Gaussian Random Fields on a shared set of query points.
    
#     Args:
#         GRFparams: Tuple (f_n, mus, l, c, b)
#             - f_n: (N_samples, n_mus)
#             - mus: (N_samples, n_mus, d)
#             - l: (N_samples,d,)
#             - c: (N_samples,)
#             - b: (N_samples,)
#         x: (N_points, d)
    
#     Returns:
#         GRF evaluations: (N_samples, N_points)
#     """
#     f_n, mus, l, c, b = GRFparams
#     N_samples, n_mus, d = mus.shape
#     N_points = x.shape[0]
#     # Compute differences: (N_samples, N_points, n_mus, d)
#     diff = (x[None, :, None, :] - mus[:, None, :, :]) / l[:, None, None, :]
#     # Basis functions: (N_samples, N_points, n_mus)
#     phi_n = np.exp(-0.5 * np.sum(diff**2, axis=-1))
#     # Weighted sum with coefficients: (N_samples, N_points)
#     f = np.einsum("Nn,Nqn->Nq", f_n, phi_n, optimize=True)
#     # Scale and shift
#     return c[:, None] * f + b[:, None]

# def GRFs_grad(GRFparams, x):
#     """
#     Evaluate gradient of Gaussian Random Fields at shared query points.
    
#     Args:
#         GRFparams: Tuple (f_n, mus, l, c, b)
#             - f_n: (N_samples, n_mus)
#             - mus: (N_samples, n_mus, d)
#             - l: (d,)
#             - c: (N_samples,)
#             - b: (N_samples,)
#         x: (N_points, d)
    
#     Returns:
#         Gradients: (N_samples, N_points, d)
#     """
#     f_n, mus, l, c, b = GRFparams
#     # Differences: (N_samples, N_points, n_mus, d)
#     diff = (x[None, :, None, :] - mus[:, None, :, :]) / l[:, None, None, :]
#     # Basis functions: (N_samples, N_points, n_mus)
#     phi = np.exp(-0.5 * np.sum(diff**2, axis=-1))
#     # Prefactor for derivative: (N_samples, N_points, n_mus, d)
#     prefactor = -diff  / l[:, None, None, :]
#     # Gradient contributions: (N_samples, N_points, n_mus, d)
#     grad_terms = prefactor * phi[:, :, :, None]
#     # Weighted sum: (N_samples, N_points, d)
#     grad = np.einsum("Nn,Nqni->Nqi", f_n, grad_terms, optimize=True)
#     return c[:, None, None] * grad

# def GRFs_d2dxi2(GRFparams, x):
#     """
#     Evaluate the second derivative of the Gaussian Random Field w.r.t. each dimension.
    
#     Args:
#         GRFparams: Tuple (f_n, mus, l, c, b)
#             - f_n: (N_samples, n_mus)
#             - mus: (N_samples, n_mus, d)
#             - l: (d,)
#             - c: (N_samples,)
#             - b: (N_samples,)
#         x: (N_points, d)
    
#     Returns:
#         Second derivatives: (N_samples, N_points, d)
#     """
#     f_n, mus, l, c, b = GRFparams
#     # Differences: (N_samples, N_points, n_mus, d)
#     diff = (x[None, :, None, :] - mus[:, None, :, :]) / l[:, None, None, :]
#     # Basis functions: (N_samples, N_points, n_mus)
#     phi = np.exp(-0.5 * np.sum(diff**2, axis=-1))
#     # Second derivative prefactor per dimension: (N_samples, N_points, n_mus, d)
#     prefactor = (diff**2 - 1) / l[:, None, None, :]**2
#     # Multiply by basis
#     d2_terms = prefactor * phi[:, :, :, None]
#     # Sum over mus (keep dimension)
#     d2_output = np.einsum("Nn,Nqni->Nqi", f_n, d2_terms, optimize=True)
#     # Scale by c
#     return c[:, None, None] * d2_output


import numpy as np
import scipy
import opt_einsum
<<<<<<< HEAD:trainingdata/GRF.py


def generate_GRF_parameters(N_samples, l, c_min, c_max, b_min, b_max):
    l = np.array(l) #Length scale of GRF
    d = len(l) #Dimensionality of GRF
    n_mus = int(np.ceil(1/np.prod(l))) #Number of GRF sample locations
    mus = np.random.uniform(0,1,size=(N_samples,n_mus,d))
    scaled_mus = mus / l[None, None, :]
    diff = scaled_mus[:, None, :, :] - scaled_mus[:, :, None, :]
    dist_sq = np.sum(diff**2, axis=-1)
    cov = np.exp(-0.5 * dist_sq)
    L = np.linalg.cholesky(cov)
    z = np.random.normal(size=(N_samples, n_mus))
    f = opt_einsum.contract('Nij,Nj->Ni', L, z)
    L_inv = np.linalg.inv(L)
    cov_inv = opt_einsum.contract('Nij,Njk->Nik', L_inv.transpose(0,2,1), L_inv)
    f_n = opt_einsum.contract('Nij,Nj->Ni', cov_inv, f)
    # f_n = np.linalg.solve(cov, f)          # (N, n)
    c = np.random.uniform(c_min, c_max, size=N_samples)  # Random scaling coefficients
    b = np.random.uniform(b_min, b_max, size=N_samples)  # Random translation coefficients
    GRFparams = f_n, mus, l, c, b
    return GRFparams

def GRFs_forward(GRFparams, x): #N_samples, N_points, n_mus, d
    f_n, mus, l, c, b = GRFparams
    phi_n = f_n[:,None,:]*np.exp(-1/2*np.sum(((x[None,:,None,:] - mus[:,None,:,:])/l[None,None,None,:])**2, axis=-1))
    return c[:,None]*np.sum(phi_n, axis=-1) + b[:,None]

def GRFs_grad(GRFparams, x):
    f_n, mus, l, c, b = GRFparams
    phi_n = f_n[:,None,:]*np.exp(-1/2*np.sum(((x[None,:,None,:] - mus[:,None,:,:])/l[None,None,None,:])**2, axis=-1))
    prefactor = -1/(l[None,None,None,:]**2)*(x[None,:,None,:] - mus[:,None,:,:])
    return c[:,None,None]*np.sum(prefactor*phi_n[:,:,:,None], axis=-2)

def GRFs_d2dxi2(GRFparams, x):
    f_n, mus, l, c, b = GRFparams
    phi_n = f_n[:,None,:]*np.exp(-1/2*np.sum(((x[None,:,None,:] - mus[:,None,:,:])/l[None,None,None,:])**2, axis=-1))
    prefactor = (x[None,:,None,:] - mus[:,None,:,:])**2/l[None,None,None,:]**4 - 1/l[None,None,None,:]**2
    return c[:,None,None]*np.sum(prefactor*phi_n[:,:,:,None], axis=-2)
=======
# 
import jax
import jax.numpy as jnp
from jax import jit, vmap
>>>>>>> 7a5994f88f5c79a7900afdf8cc33f2d3149fd82d:src/ngo/trainingdata/GRF.py

class GRF:
    def __init__(self, N_samples, l):
        super().__init__()
        self.N_samples = N_samples #Number of GRF samples
        self.l = np.array(l) #Length scale of GRF
        self.d = len(l) #Dimensionality of GRF
        self.n_mus = int(np.ceil(1/np.prod(self.l))) #Number of GRF sample locations
        self.mus = self.compute_mus()
        self.f_hat = self.compute_coeffs()

    #Sample n_mus random points on the unit square
    def compute_mus(self):
        mus = np.random.uniform(0,1,size=(self.N_samples,self.n_mus,self.d))
        return mus

    #Compute Gaussian covariance matrix cov between points
    def compute_cov(self):
        scaled_mus = self.mus / self.l[None, None, :]
        diff = scaled_mus[:, None, :, :] - scaled_mus[:, :, None, :]
        dist_sq = np.sum(diff**2, axis=-1)
        return np.exp(-0.5 * dist_sq)
    
    def compute_coeffs(self):
        cov = self.compute_cov()
        L = np.linalg.cholesky(cov)
        z = np.random.normal(size=(self.N_samples, self.n_mus))
        f = opt_einsum.contract('Nij,Nj->Ni', L, z)
        L_inv = np.linalg.inv(L)
        cov_inv = opt_einsum.contract('Nij,Njk->Nik', L_inv.transpose(0,2,1), L_inv)
        f_hat = opt_einsum.contract('Nij,Nj->Ni', cov_inv, f)
        return f_hat

    def phi_n(self, i, x):
        output = self.f_hat[i,None,:]*np.exp(-1/2*np.sum(((x[:,None,:] - self.mus[i,:,:])/self.l[None,None,:])**2, axis=-1))
        return output

    #Forward evaluation of RBF interpolated GRF
    def forward(self, i):
        def function(x):
            phi_n = self.phi_n(i, x)
            return np.sum(phi_n, axis=1)
        return function      

    #Pointwise forward evaluation of RBF interpolated GRF (required for Nutils)
    def forward_nutils(self, i):
        def function(x):
            phi_n = self.f_hat[i]*np.exp(-np.sum((x - self.mus)**2, axis=-1)/(2*self.l**2))
            return np.sum(phi_n)
        return function

    # Gradient of RBF interpolated GRF
    def grad(self, i):
        def function(x):
            phi_n = self.phi_n(i, x)
            prefactor = -1/(self.l[None,None,:]**2)*(x[:,None,:] - self.mus[i,:,:])
            return np.sum(prefactor*phi_n[:,:,None], axis=1)
        return function       

    # Laplacian of RBF interpolated GRF
    def d2dxi2(self, i):
        def function(x):
            phi_n = self.phi_n(i, x)
            prefactor = (x[:,None,:] - self.mus[i,:,:])**2/self.l[None,None,:]**4 - 1/self.l[None,None,:]**2
            return np.sum(prefactor*phi_n[:,:,None], axis=1)
        return function


class ScaledGRF:
    def __init__(self, N_samples, l, c, b):
        super().__init__()
        self.grf = GRF(N_samples, l)
        #Scaling and translation of GRF f'(x) = c f(x) + b
        self.c = c
        self.b = b
    
    def forward(self, i):
        def function(x):
            return self.c[i]*self.grf.forward(i)(x) + self.b[i]
        return function
    def forward_x(self, i, x):        
        return self.c[i]*self.grf.forward_x(i,x) + self.b[i]
        
    
    def forward_nutils(self, i):
        def function(x):
            return self.c[i]*self.grf.forward_nutils(i)(x) + self.b[i]
        return function
    
    def grad(self, i):
        def function(x):
            return self.c[i]*self.grf.grad(i)(x)
        return function
    
    def grad_x(self, i , x ):
        return self.c[i]*self.grf.grad_x(i,x)
    
    def d2dxi2(self, i):
        def function(x):
            return self.c[i]*self.grf.d2dxi2(i)(x)
        return function

    def d2dxi2_x(self, i,x):
        return self.c[i]*self.grf.d2dxi2_x(i,x)