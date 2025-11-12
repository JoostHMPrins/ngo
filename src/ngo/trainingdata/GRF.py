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
    """Gaussian Random Field generator with squared exponential covariance.
    
    Generates random fields with spatial correlation determined by length scales.
    Uses Kronecker-product structure to efficiently sample and compute the
    conditional mean of the GRF given the samples. The random fields are interpolated
    on a Gaussian Radial Basis Function (GRBF) basis on a regular grid.
    
    The covariance kernel is a product of 1D squared exponential kernels:
        K(x, x') = exp(-0.5 * sum_i ((x_i - x'_i) / l_i)^2)
    
    Attributes:
        device (torch.device): Device for tensor operations.
        dtype (torch.dtype): Data type for tensors.
        N_samples (int): Number of GRF samples to generate.
        l (torch.Tensor): Length scales for each dimension, shape (d,).
        l_Dx_ratio (int): Ratio determining grid resolution relative to length scales.
        n (tuple): Grid sizes for each dimension. "n_i = ceil(1 / l_i) + 1".
        mus (tuple): Grid points for each dimension, each element shape (n_i,).
        f_m (torch.Tensor): Basis interpolation coefficient tensor, shape (N_samples, n_0, ..., n_{d-1}).
        f (torch.Tensor): GRF samples, shape (N_samples, n_0, ..., n_{d-1}).
    """
    def __init__(self, N_samples, l, n, device, dtype):
        """Initialize the GRF generator.
        
        Parameters:
            N_samples (int): Number of random field samples to generate.
            l (tuple): Length scales for each spatial dimension. Tuple length
                determines the dimensionality of the random field.
            device (torch.device or str): Device for computation ('cpu' or 'cuda').
            dtype (torch.dtype): Data type for tensors (e.g., torch.float32).
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.N_samples = N_samples
        self.l = torch.tensor(l, device=self.device, dtype=dtype)
        self.n = n  #Number of points in each dimension
        if n==None:
            self.l_Dx_ratio = 1
            self.n = tuple((self.l_Dx_ratio*(1.0 / self.l).ceil().long() + 1).tolist())
        self.mus = self.get_mus()
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
        for i in range(len(self.l)):
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
        for i in range(len(self.l)):
            scaled_mus = self.mus[i] / self.l[i]
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
        # Forward solve
        L_i_T = tuple(L.T for L in L_i)
        y = kron_solve_triangular(L_i_T, self.f, upper=True, left=False)
        #Backward solve
        f_m = kron_solve_triangular(L_i, y, upper=False, left=False)
        return f_m