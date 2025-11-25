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
from ngo.numericalanalysis.basisfunctions import GaussianRBFBasis, TensorProductBasis


class GRF:
    """Gaussian Random Field (GRF) generator with Kronecker-product covariance structure.
    Generates samples of a Gaussian Random Field (GRF) with a covariance structure
    defined as a Kronecker product of 1D squared exponential covariance matrices.
    Uses Cholesky factorization for efficient sampling and interpolates the GRF
    on a Gaussian Radial Basis Function (GRBF) basis.
    """
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
        self.lambdas = lambdas
        self.d = self.lambdas.shape[-1]
        self.n = n  #Number of grid points in each dimension
        if n==None:
            self.l_Dx_ratio = 5 #Number of points per length scale
            self.n = tuple((self.l_Dx_ratio*(1.0 / self.lambdas).ceil().long() + 1).tolist()) #Number of grid points in each dimension
        self.N = int(torch.prod(torch.tensor(self.n, dtype=self.dtype, device=self.device)))  #Total number of grid points
        self.b = torch.tensor(b, dtype=self.dtype, device=self.device)
        self.c = torch.tensor(c, dtype=self.dtype, device=self.device)
        # Add batch dimension to b and c if missing
        if self.b.ndim == 0:
            self.b = self.b.unsqueeze(0)
        if self.c.ndim == 0:
            self.c = self.c.unsqueeze(0)
        # Generate grid points
        self.mus = self.get_mus()
        # Compute interpolation coefficients
        self.f_m = self.compute_coeffs()
        # Add batch dimension to lambdas if missing
        if self.lambdas.ndim == 1:
            self.lambdas = self.lambdas.unsqueeze(0)
    
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
        # Add jitter
        K_i = tuple(K + 1e-6*torch.eye(K.shape[0], device=self.device, dtype=self.dtype) for K in K_i)
        # Cholesky factorization
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
    

class ManufacturedSolution:
    """Manufactured solution using GRF basis functions.
    Generates a manufactured solution by sampling a Gaussian Random Field (GRF)
    and interpolating on a Gaussian Radial Basis Function (GRBF) basis.
    """
    def __init__(self, N_samples, d, lambda_min, lambda_max, b_min, b_max, c_min, c_max, device, dtype, n):
        """
        Initialize manufactured solution.

        Args:
            N_samples (int): Number of manufactured solution samples to generate.
            d (int): Number of spatial dimensions.
            lambda_min (torch.Tensor): Minimum length scales for each dimension,
                shape (d,).
            lambda_max (torch.Tensor): Maximum length scales for each dimension,
                shape (d,).
            b_min (float): Minimum mean offset for the GRF.
            b_max (float): Maximum mean offset for the GRF.
            c_min (float): Minimum standard deviation scaling for the GRF.
            c_max (float): Maximum standard deviation scaling for the GRF.
            device (torch.device): Device to perform computations on.
            dtype (torch.dtype): Data type for tensors.
            n (list or tuple): Number of grid points in each dimension,
                shape (d,).
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.N_samples = N_samples
        self.d = d
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        # Generate random length scales, b, c for each sample (unless min==max)
        self.lambdas = lambda_min if torch.all(lambda_min==lambda_max) else torch.rand((N_samples, len(self.lambda_min)), device=self.device, dtype=self.dtype)*(self.lambda_max - self.lambda_min) + self.lambda_min
        self.b = b_min*torch.ones(self.N_samples, device=self.device, dtype=self.dtype) if b_min==b_max else torch.rand(N_samples, device=self.device, dtype=self.dtype)*(b_max - b_min) + b_min
        self.c = c_min*torch.ones(self.N_samples, device=self.device, dtype=self.dtype) if c_min==c_max else torch.rand(N_samples, device=self.device, dtype=self.dtype)*(c_max - c_min) + c_min
        self.n = n
        # Generate GRF coefficients
        self.u_m = self.get_grf_coeffs()
        # Add batch dimension to lambdas if missing
        if self.lambdas.ndim == 1:
            self.lambdas = self.lambdas.unsqueeze(0)
        self.bases = ()
        # Construct TensorProductBasis from GaussianRBFBasis in each dimension
        for i in range(self.d):
            self.bases += (GaussianRBFBasis(self.lambdas[:,i], self.mus[i]),)
        self.basis = TensorProductBasis(self.bases)

    def get_grf_coeffs(self):
        """
        Generate GRF coefficients for manufactured solution.
        """
        if torch.all(self.lambda_min==self.lambda_max):
            grf = GRF(N_samples=self.N_samples, lambdas=self.lambdas, b=self.b, c=self.c, n=None, device=self.device, dtype=self.dtype)
            f_m_set = grf.f_m
        else:
            f_m_set = torch.zeros((self.N_samples, *self.n), device=self.device, dtype=self.dtype)
            for i in range(self.N_samples):
                grf = GRF(N_samples=1, lambdas=self.lambdas[i], b=self.b[i], c=self.c[i], device=self.device, dtype=self.dtype, n=self.n)
                f_m_set[i] = grf.f_m
        self.mus = grf.mus
        return f_m_set
    
    def forward(self, X):
        """"
        Evaluate manufactured solution at points X.
        
        Args:
            X (tuple): Input coordinates for each dimension, each element a 1D tensor
                of shape (Q_i,).

        Returns:
            torch.Tensor: Evaluated manufactured solution at points X, shape
                (B, Q_0, Q_1, ..., Q_{d-1}).
        """
        return self.basis.forward(self.u_m, X)
    
    def grad(self, X):
        """"
        Evaluate gradient of manufactured solution at points X.
        
        Args:
            X (tuple): Input coordinates for each dimension, each element a 1D tensor
                of shape (Q_i,).
        
        Returns:
            torch.Tensor: Evaluated gradient of manufactured solution at points X, shape
                (B, Q_0, Q_1, ..., Q_{d-1}, d).
        """
        return self.basis.grad(self.u_m, X)
    
    def laplacian(self, X):
        """"
        Evaluate laplacian of manufactured solution at points X.
        
        Args:
            X (tuple): Input coordinates for each dimension, each element a 1D tensor
                of shape (Q_i,).
        
        Returns:
            torch.Tensor: Evaluated d laplacian terms of manufactured solution at points X, shape
                (B, Q_0, Q_1, ..., Q_{d-1}, d).
        """
        return self.basis.laplacian(self.u_m, X)