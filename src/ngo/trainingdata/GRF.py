# Copyright 2025 Joost Prins
"""Gaussian Random Field (GRF) generator for training data.

This module implements a Gaussian Random Field sampler using Kronecker-product
structure for efficient generation of spatially correlated random fields in
multiple dimensions.
"""

# 3rd Party
import torch

#Local
from ngo.numericalanalysis.linalg import kron_matvec, kron_solve_triangular


class GRF:

    def __init__(self, N_samples, lambdas, b, c, device, dtype, n=None):
        """Gaussian Random Field (GRF) generator with Kronecker-product covariance structure.

        Args:
            N_samples (int): Number of GRF samples to generate.
            lambdas (list or torch.Tensor): Length scales for each spatial dimension,
                shape (d,).
            b (float or torch.Tensor): Mean offset(s) for the GRF, shape (,) or (N_samples,).
            c (float or torch.Tensor): Standard deviation scaling(s) for the GRF,
                shape (,) or (N_samples,).
            device (torch.device): Device to perform computations on.
            dtype (torch.dtype): Data type for tensors.
            n (list or tuple, optional): Number of grid points in each dimension,
                shape (d,). If None, it is determined from length scales.
        """
        super().__init__()
        """
        Initialize GRF generator.
        """
        self.device = device
        self.dtype = dtype
        self.N_samples = N_samples
        self.lambdas = torch.tensor(lambdas, device=self.device, dtype=dtype)
        self.d = self.lambdas.shape[-1]
        self.n = n  #Number of grid points in each dimension
        if n==None:
            self.l_Dx_ratio = 1 #Number of points per length scale
            self.n = tuple((self.l_Dx_ratio*(1.0 / self.lambdas).ceil().long() + 1).tolist()) #Number of grid points in each dimension
        self.N = int(torch.prod(torch.tensor(self.n)))
        self.b = torch.tensor(b, device=self.device, dtype=dtype)
        self.c = torch.tensor(c, device=self.device, dtype=dtype)
        # Add batch dimension to b and c if missing
        if self.b.ndim == 0:
            self.b = self.b.unsqueeze(0)
        if self.c.ndim == 0:
            self.c = self.c.unsqueeze(0)
        # Generate grid points
        self.mus = self.get_mus()
        # Compute interpolation coefficients
        self.f_m = self.compute_coeffs()
    
    def get_mus(self):
        """Generate grid points for each dimension.
        
        Creates uniform grids in [0, 1] for each spatial dimension with
        resolution determined by the length scale and l_Dx_ratio.
        
        Returns:
            tuple: Grid points for each dimension, each element a 1D tensor
                of shape (n_i,).
        """
        mus = ()
        for i in range(self.d):
            mus += (torch.linspace(0,1, self.n[i], device=self.device, dtype=self.dtype),)
        return mus
    
    def compute_covariances(self):
        """Compute covariance matrices for each dimension.
        
        Computes the 1D squared exponential covariance matrices that form
        the Kronecker-product structure of the full covariance.
        
        Returns:
            tuple: Covariance matrices K_i, each with shape (n_i, n_i).
        """
        K_i = ()
        for i in range(self.d):
            scaled_mus = self.mus[i] / self.lambdas[i]
            diff = scaled_mus[:, None] - scaled_mus[None, :]
            dist_sq = diff**2
            K_i += (torch.exp(-0.5 * dist_sq),)
        return K_i
    
    def compute_coeffs(self):
        """Generate GRF samples and interpolate on a Gaussian Radial Basis Function (GRBF) basis.
        
        Uses Cholesky factorization of the Kronecker-product covariance to
        generate samples, then interpolates the GRF data points f on a Gaussian
        Radial Basis Function (GRBF) basis, by solving a sequence of 1D triangular systems.
        
        Returns:
            torch.Tensor: Conditional mean of the GRF samples, shape
                (N_samples, n_0, n_1, ..., n_{d-1}).
        
        Notes:
            - Stores the GRF samples in self.f
            - The interpolation coefficients f_m satisfy:
              (K_0 ⊗ K_1 ⊗ ... ⊗ K_{d-1}) @ f_m = f
        """
        K_i = self.compute_covariances()
        L_i = tuple(torch.linalg.cholesky(K) for K in K_i)
        # Sample standard normal variables
        Z = torch.randn(size=(self.N_samples,*self.n), device=self.device, dtype=self.dtype)
        # GRF generation
        L_i_batch = tuple(L[None,:] for L in L_i)
        self.f = kron_matvec(L_i_batch, Z)
        # Scale by b and c
        self.f = self.f.reshape(self.N_samples, self.N)
        self.f = self.c[:,None]*self.f + self.b[:,None]
        self.f = self.f.reshape(self.N_samples, *self.n)
        # Forward solve
        L_i_T = tuple(L.T for L in L_i)
        y = kron_solve_triangular(L_i_T, self.f, upper=True, left=False)
        #Backward solve
        f_m = kron_solve_triangular(L_i, y, upper=False, left=False)
        return f_m