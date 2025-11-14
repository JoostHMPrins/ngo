# Copyright 2025 Joost Prins

# 3rd Party
import matplotlib.pyplot as plt
import numpy as np
import opt_einsum
from scipy.interpolate import BSpline
import torch

# Local
from ngo.ml.customlayers import discretize_functions


class GaussianRBF:
    """
    A class to represent 1D Gaussian Radial Basis Functions (RBFs).

    Attributes:
        lambdas (torch.Tensor): Length scales for the Gaussian RBFs, shape (B,) (B is batch size).
        mus (torch.Tensor): Centers of the Gaussian RBFs, shape (n,).
        n (int): Number of basis functions.
    """
    def __init__(self, lambdas, mus):
        """
        Initialize the Gaussian RBF basis.
        Args:
            lambdas (torch.Tensor): Length scales for the Gaussian RBFs, shape (B,) (B is batch size).
            mus (torch.Tensor): Centers of the Gaussian RBFs, shape (n,).
        """
        self.lambdas = lambdas
        self.mus = mus
        # Number of basis functions
        self.n = mus.shape[0]
    
    def forward(self, x):
        """
        Compute the Gaussian basis values at given points.

        Args:
            x (np.ndarray): 1D array of evaluation points, shape (n_points,).
        
        Returns:
            np.ndarray: B matrices of basis values, shape (B,n,n_points).     
        """
        basis_values = torch.exp(-0.5*((x[None,None,:] - self.mus[None,:,None]) / self.lambdas[:,None,None])**2)

        return basis_values
    
    def grad(self, x):
        """
        Compute gradients of each Gaussian basis function w.r.t. x.

        Args:
            x (np.ndarray): 1D array of evaluation points, shape (n_points,).

        Returns:
            np.ndarray: B matrices of basis gradients, shape (B,n,n_points).
        """
        # basis_gradients = -((x[None,None, None, :]  - self.mus[None,:, :, None]) / (self.lambdas[:,None,None,None] ** 2)) * self.forward(x)
        basis_gradients = -((x[None,None,:] - self.mus[None,:,None]) / (self.lambdas[:,None,None]**2)) * self.forward(x)
        return basis_gradients
    
    def laplacian(self, x):
        """
        Compute second derivative (Laplacian) of each Gaussian basis function w.r.t. x.

        Args:
            x (np.ndarray): 1D array of evaluation points, shape (n_points,).

        Returns:
            np.ndarray: B matrices of basis second derivatives, shape (B,n,n_points).
        """
        # basis_laplacians = (((x[None,None, None, :] - self.mus[None,:, :, None]) ** 2) / (self.lambdas[:,None,None,None] ** 4) - 1 / (self.lambdas[:,None,None,None] ** 2)) * self.forward(x)
        basis_laplacians = (((x[None,None,:] - self.mus[None,:,None])**2)/self.lambdas[:,None,None]**4 - 1/self.lambdas[:,None,None]**2) * self.forward(x)
        return basis_laplacians


class BSpline:
    """
    B-spline basis functions with vectorized evaluation and differentiation using PyTorch.
    """
    def __init__(self, n, p, C, dtype, device):
        """""
        Initialize the B-spline basis.

        Args:
            n: Number of basis functions.
            p: Degree of the B-splines.
            C: Continuity at internal knots (C < p).
            dtype: Torch data type for computations.
            device: Torch device for computations.
        """
        self.dtype = dtype
        self.device = device
        self.n = n
        self.p = p
        self.C = C
        self.knot_vector = torch.zeros(self.p+1, dtype=self.dtype, device=self.device)
        self.knot_vector = torch.cat((
            self.knot_vector,
            torch_repeat(
                torch.linspace(
                    0, 1,
                    int((self.n - self.p - 1)/(self.p - self.C)) + 2,
                    dtype=self.dtype, device=self.device
                )[1:-1],
                self.p - self.C
            )
        ))
        self.knot_vector = torch.cat((self.knot_vector, torch.ones(self.p+1, dtype=self.dtype, device=self.device)))

    def _basis_dp(self, x: torch.Tensor, return_prev: bool = False):
        """
        Vectorized Cox–de Boor evaluation for all basis functions in parallel.

        Args:
            x: 1D tensor of evaluation points, shape (M,).
            return_prev: If True, also returns N_{*,p-1}(x) for derivative.

        Returns:
            N_p: (n, M) values of N_{i,p}(x).
            N_pm1: (n, M) values of N_{i,p-1}(x) if return_prev=True, else None.
        """
        t = self.knot_vector
        x = x.to(dtype=self.dtype, device=self.device)
        n, p = self.n, self.p

        # Degree-0 indicator functions for all i in parallel
        t_i = t[:n].unsqueeze(1)       # (n,1)
        t_ip1 = t[1:n+1].unsqueeze(1)  # (n,1)
        N = ((t_i <= x) & (x <= t_ip1)).to(self.dtype)  # (n,M)

        if p == 0:
            return (N, None) if return_prev else N

        # Iteratively raise degree to p
        for r in range(1, p+1):
            den1 = (t[r:r+n] - t[:n]).unsqueeze(1)          # (n,1)
            den2 = (t[r+1:r+1+n] - t[1:n+1]).unsqueeze(1)   # (n,1)

            # Shift N for i+1 term
            N_ip1 = torch.cat([N[1:], torch.zeros_like(N[:1])], dim=0)

            # Weights with safe division
            w1 = torch.where(den1 > 0, (x - t[:n].unsqueeze(1)) / den1, torch.zeros_like(N))
            w2 = torch.where(den2 > 0, (t[r+1:r+1+n].unsqueeze(1) - x) / den2, torch.zeros_like(N))

            if return_prev and r == p:
                N_prev = N  # save N_{*,p-1}
            N = w1 * N + w2 * N_ip1

        return (N, N_prev) if return_prev else N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all B-spline basis functions N_{i,p}(x) in parallel.

        Args:
            x: 1D tensor of points, shape (M,).

        Returns:
            (n, M) tensor with N_{i,p}(x).
        """
        return self._basis_dp(x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate derivatives d/dx N_{i,p}(x) for all i in parallel.

        Args:
            x: 1D tensor of points, shape (M,).

        Returns:
            (n, M) tensor with d/dx N_{i,p}(x).
        """
        N_p, N_pm1 = self._basis_dp(x, return_prev=True)
        n, p = self.n, self.p
        if p == 0:
            return torch.zeros_like(N_p)

        t = self.knot_vector
        den1 = (t[p:p+n] - t[:n]).unsqueeze(1)          # (n,1)
        den2 = (t[p+1:p+1+n] - t[1:n+1]).unsqueeze(1)   # (n,1)

        term1 = torch.where(den1 > 0, (p / den1) * N_pm1, torch.zeros_like(N_pm1))
        N_pm1_ip1 = torch.cat([N_pm1[1:], torch.zeros_like(N_pm1[:1])], dim=0)
        term2 = torch.where(den2 > 0, (p / den2) * N_pm1_ip1, torch.zeros_like(N_pm1))

        return term1 - term2


class ChebyshevTBasis1D:
    """
    A class to represent 1D Chebyshev polynomial basis functions of the first kind.

    Attributes:
        n (int): Number of basis functions.
    """
    def __init__(self, n):
        """
        Initialize the Chebyshev basis.

        Args:
            n (int): Number of basis functions.
        """
        self.n = n

    def basis_function(self, n):
        """
        Compute the nth Chebyshev basis function.

        Args:
            n (int): Index of the basis function.

        Returns:
            np.polynomial.chebyshev.Chebyshev: The nth Chebyshev basis function.
        """
        c = np.zeros(self.n)
        c[n] = 1
        basisfunction = np.polynomial.chebyshev.Chebyshev(coef=c,domain=[0,1])
        return basisfunction
    
    def forward(self, x):
        """
        Compute the Chebyshev basis values at given points.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Basis values at the input points. Shape: (n_points, n).
        """
        basis_values = np.zeros((x.shape[0],self.n))
        for n in range(self.n):
            basis_values[:, n] = self.basis_function(n)(x)
        return basis_values
    
    def basis_gradient(self, n):
        """
        Compute the gradient of the nth Chebyshev basis function.

        Args:
            n (int): Index of the basis function.

        Returns:
            np.polynomial.chebyshev.Chebyshev: Derivative of the nth basis function.
        """
        basisfunction = self.basis_function(n)
        derivative = basisfunction.deriv(m=1)
        return derivative
    
    def grad(self, x):
        """
        Compute the gradients of the Chebyshev basis functions.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Gradients of the basis functions at the input points. Shape: (n_points, n).
        """
        basis_gradients = np.zeros((x.shape[0],self.n))
        for n in range(self.n):
            basis_grad = self.basis_gradient(n)(x)
            basis_gradients[:, n] = basis_grad
        return basis_gradients


class SincBasis1D:
    """
    A class to represent 1D sinc basis functions (sin(x)/x).

    Attributes:
        n (int): Number of basis functions.
        Dx (float): Spacing between grid points.
        grid (np.ndarray): Grid points for the basis functions. Shape: (n,).
    """

    def __init__(self, n):
        """
        Initialize the sinc basis.

        Args:
            n (int): Number of basis functions.
        """
        self.n = n
        self.Dx = 1/(n-1)
        self.grid = np.linspace(0,1,self.n)

    def sinc(self, x):
        """
        Compute the sinc function values.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Sinc function values at the input points. Shape: (n_points,).
        """
        output = np.sin(x)/x
        output[x==0] = 1
        return output
    
    def dsincdx(self, x):
        """
        Compute the derivative of the sinc function.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Derivative of the sinc function at the input points. Shape: (n_points,).
        """
        output = np.cos(x)/x - np.sin(x)/x**2
        output[x==0] = 0
        return output
    
    def forward(self, x):
        """
        Compute the sinc basis values at given points.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Basis values at the input points. Shape: (n_points, n).
        """
        basis_values = np.zeros((x.shape[0],self.n))
        for n in range(self.n):
            x_scaled = np.pi*(x - self.grid[n])/self.Dx
            basis_values[:, n] = self.sinc(x_scaled)
        return basis_values
    
    def grad(self, x):
        """
        Compute the gradients of the sinc basis functions.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Gradients of the basis functions at the input points. Shape: (n_points, n).
        """
        basis_gradients = np.zeros((x.shape[0],self.n))
        for n in range(self.n):
            x_scaled = np.pi*(x - self.grid[n])/self.Dx
            basis_gradients[:, n] = np.pi/self.Dx*self.dsincdx(x_scaled)
        return basis_gradients


class PolynomialBasis1D:
    """
    A class to represent 1D polynomial basis functions.
    The basis functions are defined as:
    phi_0 = c_0, phi_1 = c_1 * x, phi_2 = c_2 * x^2, ..., phi_n = c_n * x^n.

    Attributes:
        n (int): Number of basis functions.
        exponents (np.ndarray): Array of the exponents of the individual basis functions.
    """
    def __init__(self, n):
        """
        Initialize the polynomial basis.

        Args:
            n (int): Number of basis functions.
        """
        self.n = n
        self.exponents = np.arange(0,self.n)
    
    def forward(self, x):
        """
        Compute the polynomial basis values at given points.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Basis values at the input points. Shape: (n_points, n).
        """
        basis_values = x[:,None]**self.exponents[None,:]
        return basis_values
    
    def grad(self, x):
        """
        Compute the gradients of the polynomial basis functions.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Gradients of the basis functions at the input points. Shape: (n_points, n).
        """
        basis_gradients = self.exponents[None,:]*x[:,None]**(self.exponents-1)
        basis_gradients[:,0] = 0
        return basis_gradients


class FourierBasis1D:
    """
    A class to represent 1D Fourier basis functions in sine-cosine form.
    The Fourier basis functions are defined as:
    phi_0 = 1, phi_1 = cos(k_1 * x), phi_2 = sin(k_1 * x), ..., 
    alternating between cosine and sine terms.

    Attributes:
        n (int): Number of basis functions.
        n (np.ndarray): Array of indices for the basis functions.
        L (float): Length of the domain (should be unequal to 1 to allow for aperiodicity).
        k_n (np.ndarray): Wave numbers for the basis functions.
    """
    def __init__(self, n):
        """
        Initialize the Fourier basis.

        Args:
            n (int): Number of basis functions.
        """
        self.n = n
        self.n = np.arange(0,self.n)
        self.L = 1.2
        self.k_n = 2*np.pi*self.n/self.L
    
    def forward(self, x):
        """
        Compute the Fourier basis values at given points.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Basis values at the input points. Shape: (n_points, n).
        """
        basis_values = np.zeros((x.shape[0],self.n))
        n = 0
        for i in np.arange(0,self.n,2):
            basis_values[:,i] = np.cos(self.k_n[n]*x)
            n+=1
        n = 1
        for i in np.arange(1,self.n,2):
            basis_values[:,i] = np.sin(self.k_n[n]*x)
            n+=1
        return basis_values
    
    def grad(self, x):
        """
        Compute the gradients of the Fourier basis functions.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Gradients of the basis functions at the input points. Shape: (n_points, n).
        """
        basis_gradients = np.zeros((x.shape[0],self.n))
        n = 0
        for i in np.arange(0,self.n,2):
            basis_gradients[:,i] = -self.k_n[n]*np.sin(self.k_n[n]*x)
            n+=1
        n = 1
        for i in np.arange(1,self.n,2):
            basis_gradients[:,i] = self.k_n[n]*np.cos(self.k_n[n]*x)
            n+=1
        return basis_gradients


class TensorProductBasis:
    """
    Tensor-product (Kronecker) basis built from multiple 1D basis objects.

    This class combines several 1D basis objects into a d-dimensional basis by
    taking tensor products of the 1D basis functions. It evaluates a field
    given modal coefficients `u_m` at tensor-product evaluation points and
    computes partial derivatives by replacing a 1D basis with its derivative
    in the corresponding dimension.

    Requirements for each 1D basis object:
      - .n attribute: number of basis functions in that 1D basis
      - .forward(xi): returns shape (n_basis_i, n_points_i)
      - .grad(xi): returns shape (n_basis_i, n_points_i)

    Attributes:
        bases (tuple): Sequence of 1D basis objects.
        d (int): Number of dimensions (len(bases)).
        n (tuple): Number of basis functions per dimension (n1, ..., nd).
    """
    def __init__(self, bases):
        """
        Initialize the tensorized basis.

        Args:
            bases (list): A list of 1D basis objects.
        """
        self.bases = bases
        self.d = len(bases)
        self.n = ()
        for i in range(len(bases)):
            self.n += (bases[i].n,)

    def Phi(self, X):
        """
        Evaluate the 1D basis functions at provided points.
        
        Args:
            X (tuple/list): Sequence of point arrays for each dimension.
                            X[i] has shape (n_points_i,).
        Returns:
            tuple: Sequence of basis evaluation matrices for each dimension.
                   Each element has shape (n_basis_i, n_points_i).
        """
        # Construct basis evaluations
        Phi = ()
        # For each dimension, compute basis evaluation matrix
        for i in range(self.d):
            Phi += (self.bases[i].forward(X[i]),)
        # Add batch dimension if missing
        Phi = tuple(Phi_i[None,:,:] if Phi_i.ndim==2 else Phi_i for Phi_i in Phi)
        return Phi
    
    def dPhi(self, X):
        """
        Evaluate the derivatives of the 1D basis functions at provided points.

        Args:
            X (tuple/list): Sequence of point arrays for each dimension.
                            X[i] has shape (n_points_i,).
        Returns:
            tuple: Sequence of basis derivative matrices for each dimension.
                   Each element has shape (n_basis_i, n_points_i).
        """
        # Construct basis derivatives
        dPhi = ()
        # For each dimension, compute basis derivative matrix
        for i in range(self.d):
            dPhi += (self.bases[i].grad(X[i]),)
        # Add batch dimension if missing
        dPhi = tuple(dPhi_i[None,:,:] if dPhi_i.ndim==2 else dPhi_i for dPhi_i in dPhi)
        return dPhi
    
    def ddPhi(self, X):
        """
        Evaluate the second derivatives of the 1D basis functions at provided points.

        Args:
            X (tuple/list): Sequence of point arrays for each dimension.
                            X[i] has shape (n_points_i,).
        Returns:
            tuple: Sequence of basis second derivative matrices for each dimension.
                   Each element has shape (n_basis_i, n_points_i).
        """
        # Construct basis second derivatives
        ddPhi = ()
        # For each dimension, compute basis second derivative matrix
        for i in range(self.d):
            ddPhi += (self.bases[i].laplacian(X[i]),)
        # Add batch dimension if missing
        ddPhi = tuple(ddPhi_i[None,:,:] if ddPhi_i.ndim==2 else ddPhi_i for ddPhi_i in ddPhi)
        return ddPhi

    def gradPhi_i(self, X, i):
        """
        Evaluate the derivatives of the 1D basis functions in dimension i at provided points.

        Args:
            X (tuple/list): Sequence of point arrays for each dimension.
                            X[i] has shape (n_points_i,).
            i (int): Dimension index for which to compute the basis derivatives.
        
        Returns:
            tuple: Sequence of basis evaluation matrices for each dimension,
                   with the i-th element replaced by its derivative.
                   Each element has shape (n_basis_i, n_points_i).
        """
        # Compute basis evaluation matrices
        Phi = self.Phi(X)
        # Compute basis derivative matrix in dimension i
        dPhi_i = self.bases[i].grad(X[i])
        # Define gradPhi_i tuple as list to allow modification
        gradPhi_i = list(Phi)
        # Replace basis evaluation in dimension i with its derivative
        gradPhi_i[i] = dPhi_i
        # Convert back to tuple
        gradPhi_i = tuple(gradPhi_i)
        return gradPhi_i
    
    def laplacianPhi_i(self, X, i):
        """
        Evaluate the second derivatives of the 1D basis functions in dimension i at provided points.

        Args:
            X (tuple/list): Sequence of point arrays for each dimension.
                            X[i] has shape (n_points_i,).
            i (int): Dimension index for which to compute the basis second derivatives.

        Returns:
            tuple: Sequence of basis evaluation matrices for each dimension,
                   with the i-th element replaced by its second derivative.
                   Each element has shape (n_basis_i, n_points_i).
        """
        # Compute basis evaluation matrices
        Phi = self.Phi(X)
        # Compute basis second derivative matrix in dimension i
        laplacianPhi_i = self.bases[i].laplacian(X[i])
        # Define laplacianPhi_i tuple as list to allow modification
        laplacianPhi_i = list(Phi)
        # Replace basis evaluation in dimension i with its second derivative
        laplacianPhi_i[i] = laplacianPhi_i
        # Convert back to tuple
        laplacianPhi_i = tuple(laplacianPhi_i)
        return laplacianPhi_i

    def forward(self, u_m, X):
        """
        Evaluate the tensor-product expansion at provided points.

        Args:
            u_m (np.ndarray): Modal coefficients with shape (N_samples, prod(n)) or
                              (N_samples, n1, n2, ..., nd). N_samples is batch size.
            X (tuple/list): Sequence of point arrays for each dimension.
                            X[i] has shape (n_points_i,).

        Returns:
            np.ndarray: Evaluated values at tensor-product grid points,
                        shape (N, n_points_1, ..., n_points_d).
        """
        # reshape coefficients into (N_samples, n1, n2, ..., nd)
        u_q = u_m.reshape(u_m.shape[0],*self.n)
        # Compute basis evaluation matrices
        Phi = self.Phi(X)
        # sequentially contract coefficient tensor with each 1D basis evaluation
        for i in range(self.d):
            # contract along the i-th dimension
            u_q = u_q.moveaxis(1, -1)  # move basis axis to the end for einsum
            u_q = torch.einsum('N...n,Nnq->N...q', u_q, Phi[i])
        # final shape is (N_samples, n_points_1, ..., n_points_d)
        return u_q

    def grad(self, u_m, X):
        """
        Compute partial derivatives of the tensor-product expansion.
        For each spatial dimension i the method replaces the 1D basis eval in
        that dimension with its derivative and contracts as in `forward`.

        Args:
            u_m (np.ndarray): Modal coefficients, shape (N, prod(n)) or (N, n1, ..., nd).
            X (tuple/list): Sequence of point arrays for each dimension.
                            x[i] has shape (n_points_i,).
        Returns:
            np.ndarray: Gradients at tensor-product points with shape
                        (N, n_points_1, ..., n_points_d, d). The last axis indexes
                        partial derivatives w.r.t. each coordinate.
        """
        #Compute function evaluation and gradient matrices
        Phi = self.Phi(X)
        dPhi = self.dPhi(X)
        # Compute derivative for each dimension by contracting with dPhi in that dim.
        gradu_q = ()
        for i in range(self.d):
            # start from coefficients reshaped to (N, n1, ..., nd)
            u_q = u_m.reshape(u_m.shape[0],*self.n)
            for j in range(self.d):
                u_q = u_q.moveaxis(1, -1)  # move basis axis to the end for einsum
                if j==i:
                    # use derivative matrix in the i-th dimension
                    u_q = torch.einsum('N...n,Nnq->N...q', u_q, dPhi[j])
                else:
                    # use value matrix in other dimensions
                    u_q = torch.einsum('N...n,Nnq->N...q', u_q, Phi[j])
            # append gradient in direction i to gradu_q tuple
            gradu_q += (u_q,)
        #Reshape such that the last axis indexes the coordinate to which derivative has been taken
        gradu_q = torch.stack(gradu_q, axis=-1)
        return gradu_q
    
    def laplacian(self, u_m, X):
        """
        Compute Laplacian of the tensor-product expansion.
        For each spatial dimension i the method replaces the 1D basis eval in
        that dimension with its second derivative and contracts as in `forward`.

        Args:
            u_m (np.ndarray): Modal coefficients, shape (N, prod(n)) or (N, n1, ..., nd).
            X (tuple/list): Sequence of point arrays for each dimension.
                            x[i] has shape (n_points_i,).
        Returns:
            np.ndarray: Laplacians at tensor-product points with shape
                        (N, n_points_1, ..., n_points_d, d). The last axis indexes
                        second derivatives w.r.t. each coordinate.
        """
        #Compute function evaluation and second derivative matrices
        Phi = self.Phi(X)
        ddPhi = self.ddPhi(X)
        # Compute derivative for each dimension by contracting with ddPhi in that dim.
        lapu_q = ()
        for i in range(self.d):
            # start from coefficients reshaped to (N, n1, ..., nd)
            u_q = u_m.reshape(u_m.shape[0],*self.n)
            for j in range(self.d):
                u_q = u_q.moveaxis(1, -1)  # move basis axis to the end for einsum
                if j==i:
                    # use second derivative matrix in the i-th dimension
                    u_q = torch.einsum('N...n,Nnq->N...q', u_q, ddPhi[j])
                else:
                    # use value matrix in other dimensions
                    u_q = torch.einsum('N...n,Nnq->N...q', u_q, Phi[j])
            # append gradient in direction i to gradu_q tuple
            lapu_q += (u_q,)
        #Reshape such that the last axis indexes the coordinate to which derivative has been taken
        lapu_q = torch.stack(lapu_q, axis=-1)
        return lapu_q