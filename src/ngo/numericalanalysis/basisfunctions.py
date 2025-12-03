# Copyright 2025 Joost Prins

# 3rd Party
import torch
import opt_einsum

from scipy.interpolate import BSpline
import numpy as np

# Local
from ngo.ml.torch_utils import torch_repeat


class GaussianRBFBasis:
    """
    A class to represent 1D Gaussian Radial Basis Functions (RBFs).

    Attributes:
        lambdas (torch.Tensor): Length scales for the Gaussian RBFs, shape (B,) (B is batch size).
        mus (torch.Tensor): Centers of the Gaussian RBFs, shape (N,).
        N (int): Number of basis functions.
    """
    def __init__(self, lambdas, mus):
        """
        Initialize the Gaussian RBF basis.
        Args:
            lambdas (torch.Tensor): Length scales for the Gaussian RBFs, shape (B,) (B is batch size).
            mus (torch.Tensor): Centers of the Gaussian RBFs, shape (N,).
        """
        self.lambdas = lambdas
        self.mus = mus
        # Number of basis functions
        self.N = mus.shape[0]
    
    def forward(self, x):
        """
        Compute the Gaussian basis values at given points.

        Args:
            x (torch.tensor): 1D array of evaluation points, shape (Q,).
        
        Returns:
            torch.tensor: B matrices of basis values, shape (B,N,Q).     
        """
        basis_values = torch.exp(-0.5*((x[None,None,:] - self.mus[None,:,None]) / self.lambdas[:,None,None])**2)
        return basis_values
    
    def grad(self, x):
        """
        Compute gradients of each Gaussian basis function w.r.t. x.

        Args:
            x (torch.tensor): 1D array of evaluation points, shape (Q,).

        Returns:
            torch.tensor: B matrices of basis gradients, shape (B,N,Q).
        """
        basis_gradients = -((x[None,None,:] - self.mus[None,:,None]) / (self.lambdas[:,None,None]**2)) * self.forward(x)
        return basis_gradients
    
    def laplacian(self, x):
        """
        Compute second derivative (Laplacian) of each Gaussian basis function w.r.t. x.

        Args:
            x (torch.tensor): 1D array of evaluation points, shape (Q,).

        Returns:
            torch.tensor: B matrices of basis second derivatives, shape (B,N,Q).
        """
        basis_laplacians = (((x[None,None,:] - self.mus[None,:,None])**2)/self.lambdas[:,None,None]**4 - 1/self.lambdas[:,None,None]**2) * self.forward(x)
        return basis_laplacians


class BSplineBasis:
    """
    B-spline basis functions with vectorized evaluation and differentiation using PyTorch.
    """
    def __init__(self, N, p, C, dtype, device):
        """""
        Initialize the B-spline basis.

        Args:
            N: Number of basis functions.
            p: Degree of the B-splines.
            C: Continuity at internal knots (C < p).
            dtype: Torch data type for computations.
            device: Torch device for computations.
        """
        self.dtype = dtype
        self.device = device
        self.N = N
        self.L = 1.0  # Scaling to parent domain [0,L]
        self.Dx = 0 # Translation to parent domain if needed
        self.p = p
        self.C = C
        self.knot_vector = torch.zeros(self.p+1, dtype=self.dtype, device=self.device)
        self.knot_vector = torch.cat((
            self.knot_vector,
            torch_repeat(
                torch.linspace(
                    0, 1,
                    int((self.N - self.p - 1)/(self.p - self.C)) + 2,
                    dtype=self.dtype, device=self.device
                )[1:-1],
                self.p - self.C
            )
        ))
        self.knot_vector = torch.cat((self.knot_vector, torch.ones(self.p+1, dtype=self.dtype, device=self.device)))
        # Set tolerances based on dtype
        if self.dtype == torch.float32:
            self.rtol, self.atol = 1e-5, 1e-7
        if self.dtype == torch.float64:
            self.rtol, self.atol = 1e-13, 1e-15

    def _basis_dp(self, x: torch.Tensor, return_prev: bool = False):
        """
        Vectorized Cox–de Boor evaluation for all basis functions in parallel.

        Args:
            x: 1D tensor of evaluation points, shape (Q,).
            return_prev: If True, also returns I_{*,p-1}(x) for derivative.

        Returns:
            I_p: (N, Q) values of I_{i,p}(x).
            I_pm1: (N, Q) values of I_{i,p-1}(x) if return_prev=True, else None.
        """
        t = self.knot_vector
        x = x.to(dtype=self.dtype, device=self.device)
        N, p = self.N, self.p

        # Degree-0 indicator functions: [t_i, t_{i+1}) except last interval includes right endpoint
        t_i = t[:N].unsqueeze(1)       # (N,1)
        t_ip1 = t[1:N+1].unsqueeze(1)  # (N,1)
        
        # Use half-open intervals [t_i, t_{i+1})
        I = (t_i <= x) & (x < t_ip1)  # (N,Q) boolean
        
        # # Special case: include right endpoint for the last interval
        # I[-1] = I[-1] | (x == t_ip1[-1])

        # Special case: include right endpoint using isclose
        at_right_endpoint = torch.isclose(x, t_ip1[-1], rtol=self.rtol, atol=self.atol)
        I[-1] = I[-1] | at_right_endpoint
        
        # Convert to float after logical operations
        I = I.to(self.dtype)

        if p == 0:
            return (I, None) if return_prev else I

        # Iteratively raise degree to p
        for r in range(1, p+1):
            den1 = (t[r:r+N] - t[:N]).unsqueeze(1)          # (N,1)
            den2 = (t[r+1:r+1+N] - t[1:N+1]).unsqueeze(1)   # (N,1)

            # Shift I for i+1 term
            I_ip1 = torch.cat([I[1:], torch.zeros_like(I[:1])], dim=0)

            # Weights with safe division
            w1 = torch.where(den1 > 0, (x - t[:N].unsqueeze(1)) / den1, torch.zeros_like(I))
            w2 = torch.where(den2 > 0, (t[r+1:r+1+N].unsqueeze(1) - x) / den2, torch.zeros_like(I))

            if return_prev and r == p:
                I_prev = I  # save I_{*,p-1}
            I = w1 * I + w2 * I_ip1

        return (I, I_prev) if return_prev else I

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all B-spline basis functions I_{i,p}(x) in parallel.

        Args:
            x: 1D tensor of points, shape (Q,).

        Returns:
            (B, N, Q) tensor with I_{i,p}(x).
        """
        # Scale x to parent domain
        x = (x - self.Dx) / self.L
        # Compute basis functions
        output = self._basis_dp(x)
        # Add batch dimension
        output = output[None,:,:]
        return output
    
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate derivatives d/dx I_{i,p}(x) for all i in parallel.

        Args:
            x: 1D tensor of points, shape (Q,).

        Returns:
            (B, N, Q) tensor with d/dx I_{i,p}(x).
        """
        # Scale x to parent domain
        x = (x - self.Dx) / self.L
        
        # Compute basis function derivatives
        I_p, I_pm1 = self._basis_dp(x, return_prev=True)
        N, p = self.N, self.p
        if p == 0:
            return torch.zeros_like(I_p)

        # No need to fix I_pm1 endpoint here - it's already handled in _basis_dp
        # The issue is that _basis_dp is called recursively and handles endpoints correctly
        
        den1 = (self.knot_vector[p:p+N] - self.knot_vector[:N]).unsqueeze(1)
        den2 = (self.knot_vector[p+1:p+1+N] - self.knot_vector[1:N+1]).unsqueeze(1)

        term1 = torch.where(den1 > 0, (p / den1) * I_pm1, torch.zeros_like(I_pm1))
        I_pm1_ip1 = torch.cat([I_pm1[1:], torch.zeros_like(I_pm1[:1])], dim=0)
        term2 = torch.where(den2 > 0, (p / den2) * I_pm1_ip1, torch.zeros_like(I_pm1))

        output = term1 - term2
        output = output[None,:,:]
        # Scale derivative back to original domain
        output = output / self.L
        return output
    

class SincBasis:
    """
    A class to represent 1D sinc basis functions (sin(x)/x).
    """

    def __init__(self, N, dtype, device):
        """
        Initialize the sinc basis.

        Args:
            N (int): Number of basis functions.
            Dx (float): Spacing between grid points.
            grid (torch.tensor): Grid points for the basis functions. Shape: (n,).
            dtype (torch.dtype): Data type of the tensors.
            device (torch.device): Device on which tensors are allocated.
        """
        self.dtype = dtype
        self.device = device
        self.N = N
        self.Dx = 1/(N-1)
        self.grid = torch.linspace(0,1,self.N, dtype=dtype, device=device)

    def sinc(self, x):
        """
        Compute the sinc function values.

        Args:
            x (torch.tensor): Input points. Shape: (Q,).

        Returns:
            torch.tensor: Sinc function values at the input points. Shape: (Q,).
        """
        output = torch.sin(x)/x
        output[x==0] = 1
        return output
    
    def dsincdx(self, x):
        """
        Compute the derivative of the sinc function.

        Args:
            x (torch.tensor): Input points. Shape: (Q,).

        Returns:
            torch.tensor: Derivative of the sinc function at the input points. Shape: (Q,).
        """
        output = torch.cos(x)/x - torch.sin(x)/x**2
        output[x==0] = 0
        return output
    
    def forward(self, x):
        """
        Compute the sinc basis values at given points.

        Args:
            x (torch.tensor): Input points. Shape: (Q,).

        Returns:
            torch.tensor: Basis values at the input points. Shape: (N, Q).
        """
        basis_values = torch.zeros((self.N,x.shape[0]), dtype=self.dtype, device=self.device)
        for n in range(self.N):
            x_scaled = torch.pi*(x - self.grid[n])/self.Dx
            basis_values[n, :] = self.sinc(x_scaled)
        output = basis_values
        # Add batch dimension
        output = output[None,:,:]
        return output
    
    def grad(self, x):
        """
        Compute the gradients of the sinc basis functions.

        Args:
            x (torch.tensor): Input points. Shape: (Q,).

        Returns:
            torch.tensor: Gradients of the basis functions at the input points. Shape: (N, Q).
        """
        basis_gradients = torch.zeros((self.N,x.shape[0]), dtype=self.dtype, device=self.device)
        for n in range(self.N):
            x_scaled = torch.pi*(x - self.grid[n])/self.Dx
            basis_gradients[n, :] = torch.pi/self.Dx*self.dsincdx(x_scaled)
        output = basis_gradients
        # Add batch dimension
        output = output[None,:,:]
        return output


class PolynomialBasis:
    """
    A class to represent 1D polynomial basis functions.
    The basis functions are defined as:
    phi_0 = c_0, phi_1 = c_1 * x, phi_2 = c_2 * x^2, ..., phi_n = c_n * x^n.
    """
    def __init__(self, N, dtype, device):
        """
        Initialize the polynomial basis.
        Args:
            N (int): Number of basis functions.
            exponents (torch.tensor): Array of the exponents of the individual basis functions.
            dtype (torch.dtype): Data type of the tensors.
            device (torch.device): Device on which tensors are allocated.
        """
        self.dtype = dtype
        self.device = device
        self.N = N
        self.exponents = torch.arange(0,self.N, dtype=dtype, device=device)
    
    def forward(self, x):
        """
        Compute the polynomial basis values at given points.

        Args:
            x (torch.tensor): Input points. Shape: (Q,).

        Returns:
            torch.tensor: Basis values at the input points. Shape: (N, Q).
        """
        basis_values = x[None,:]**self.exponents[:,None]
        output = basis_values
        # Add batch dimension
        output = output[None,:,:]
        return output
    
    def grad(self, x):
        """
        Compute the gradients of the polynomial basis functions.

        Args:
            x (torch.tensor): Input points. Shape: (Q,).

        Returns:
            torch.tensor: Gradients of the basis functions at the input points. Shape: (N, Q).
        """
        basis_gradients = self.exponents[:,None]*x[None,:]**(self.exponents-1) 
        basis_gradients[0,:] = 0
        output = basis_gradients
        # Add batch dimension
        output = output[None,:,:]
        return output


class FourierBasis1D:
    """
    A class to represent 1D Fourier basis functions in sine-cosine form.
    The Fourier basis functions are defined as:
    phi_0 = 1, phi_1 = cos(k_1 * x), phi_2 = sin(k_1 * x), ..., 
    alternating between cosine and sine terms.
    """
    def __init__(self, N, dtype, device):
        """
        Initialize the Fourier basis.

        Args:
            N (int): Number of basis functions.
            n (torch.tensor): Array of indices for the basis functions.
            L (float): Length of the domain (should be unequal to 1 to allow for aperiodicity).
            k_n (torch.tensor): Wave numbers for the basis functions.
            N (int): Number of basis functions.
            dtype (torch.dtype): Data type of the tensors.
            device (torch.device): Device on which tensors are allocated.
        """
        self.dtype = dtype
        self.device = device
        self.N = N
        self.n = torch.arange(0,self.N, dtype=dtype, device=device)
        self.L = 1.2
        self.k_n = 2*torch.pi*self.n/self.L
    
    def forward(self, x):
        """
        Compute the Fourier basis values at given points.

        Args:
            x (torch.tensor): Input points. Shape: (Q,).
        Returns:
            torch.tensor: Basis values at the input points. Shape: (N, Q).
        """
        basis_values = torch.zeros((self.N,x.shape[0]), dtype=self.dtype, device=self.device)
        n = 0
        for i in torch.arange(0,self.N,2):
            basis_values[i,:] = torch.cos(self.k_n[n]*x)
            n+=1
        n = 1
        for i in torch.arange(1,self.N,2):
            basis_values[i,:] = torch.sin(self.k_n[n]*x)
            n+=1
        output = basis_values
        # Add batch dimension
        output = output[None,:,:]
        return output
    
    def grad(self, x):
        """
        Compute the gradients of the Fourier basis functions.

        Args:
            x (torch.tensor): Input points. Shape: (Q,).

        Returns:
            torch.tensor: Gradients of the basis functions at the input points. Shape: (N, Q).
        """
        basis_gradients = torch.zeros((self.N,x.shape[0]), dtype=self.dtype, device=self.device)
        n = 0
        for i in torch.arange(0,self.N,2):
            basis_gradients[i,:] = -self.k_n[n]*torch.sin(self.k_n[n]*x)
            n+=1
        n = 1
        for i in torch.arange(1,self.N,2):
            basis_gradients[i,:] = self.k_n[n]*torch.cos(self.k_n[n]*x)
            n+=1
        output = basis_gradients
        # Add batch dimension
        output = output[None,:,:]
        return output


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
      - .forward(xi): returns shape (N_i, Q_i)
      - .grad(xi): returns shape (N_i, Q_i)

    Attributes:
        bases (tuple): Sequence of 1D basis objects.
        d (int): Number of dimensions (len(bases)).
        N (tuple): Number of basis functions per dimension (N_0, ..., N_{d-1}).
    """
    def __init__(self, bases):
        """
        Initialize the tensorized basis.

        Args:
            bases (list): A list of 1D basis objects.
        """
        self.bases = bases
        self.d = len(bases)
        self.N = ()
        for i in range(len(bases)):
            self.N += (bases[i].N,)

    def Phi(self, X):
        """
        Evaluate the 1D basis functions at provided points.
        
        Args:
            X (tuple/list): Sequence of point arrays for each dimension.
                            X[i] has shape (Q_i,).
        Returns:
            tuple: Sequence of basis evaluation matrices for each dimension.
                   Each element has shape (N_i, Q_i).
        """
        # Construct basis evaluations
        Phi = ()
        # For each dimension, compute basis evaluation matrix
        for i in range(self.d):
            Phi += (self.bases[i].forward(X[i]),)
        return Phi

    def forward(self, U_m, X):
        """
        Evaluate the tensor-product expansion at provided points.

        Args:
            U_m (torch.tensor): Modal coefficients with shape (B, prod(N)) or
                              (B, N_1, N_2, ..., N_d). B is batch size.
            X (tuple/list): Sequence of point arrays for each dimension.
                            X[i] has shape (Q_i,).

        Returns:
            torch.tensor: Evaluated values at tensor-product grid points,
                        shape (B, Q_i_0, ..., Q_i_{d-1}).
        """
        # reshape coefficients into (B, N_0, N_1, ..., N_{d-1})
        u_q = U_m.reshape(U_m.shape[0],*self.N)
        # sequentially contract coefficient tensor with each 1D basis evaluation
        for i in range(self.d):
            # move basis axis to the end for einsum
            u_q = u_q.moveaxis(1, -1) 
            # Compute basis evaluation matrix in dimension i
            Phi_i = self.bases[i].forward(X[i])
            # Contract along i-th dimension
            u_q = opt_einsum.contract('B...n,Bnq->B...q', u_q, Phi_i)
        return u_q

    def grad(self, U_m, X):
        """
        Compute partial derivatives of the tensor-product expansion.
        For each spatial dimension i the method replaces the 1D basis eval in
        that dimension with its derivative and contracts as in `forward`.

        Args:
            U_m (torch.tensor): Modal coefficients, shape (B, prod(n)) or (B, N_0, ..., N_{d-1}).
            X (tuple/list): Sequence of point arrays for each dimension.
                            x[i] has shape (Q_i,).
        Returns:
            torch.tensor: Gradients at tensor-product points with shape
                        (B, Q_i_0, ..., Q_i_{d-1}, d). The last axis indexes
                        partial derivatives w.r.t. each coordinate.
        """
        # Compute derivative for each dimension by contracting with dPhi in that dim.
        gradu_q = ()
        for i in range(self.d):
            # start from coefficients reshaped to (B, N_0, ..., N_{d-1})
            u_q = U_m.reshape(U_m.shape[0],*self.N)
            for j in range(self.d):
                u_q = u_q.moveaxis(1, -1)  # move basis axis to the end for einsum
                if j==i:
                    # use derivative matrix in the i-th dimension
                    dPhi_j = self.bases[j].grad(X[j])
                    # contract along j-th dimension
                    u_q = opt_einsum.contract('B...n,Bnq->B...q', u_q, dPhi_j)
                else:
                    # use value matrix in other dimensions
                    Phi_j = self.bases[j].forward(X[j])
                    # contract along j-th dimension
                    u_q = opt_einsum.contract('B...n,Bnq->B...q', u_q, Phi_j)
            # append gradient in direction i to gradu_q tuple
            gradu_q += (u_q,)
        #Reshape such that the last axis indexes the coordinate to which derivative has been taken
        gradu_q = torch.stack(gradu_q, axis=-1)
        return gradu_q

    def laplacian(self, U_m, X):
        """
        Compute Laplacian of the tensor-product expansion.
        For each spatial dimension i the method replaces the 1D basis eval in
        that dimension with its second derivative and contracts as in `forward`.

        Args:
            U_m (torch.tensor): Modal coefficients, shape (B, prod(N)) or (B, N_0, ..., N_{d-1}).
            X (tuple/list): Sequence of point arrays for each dimension.
                            x[i] has shape (Q_i,).
        Returns:
            torch.tensor: Laplacians at tensor-product points with shape
                        (B, Q_i_0, ..., Q_i_{d-1}, d). The last axis indexes
                        second derivatives w.r.t. each coordinate.
        """
        # Compute derivative for each dimension by contracting with ddPhi in that dim.
        lapu_q = ()
        for i in range(self.d):
            # start from coefficients reshaped to (B, N_0, ..., N_{d-1})
            U_q = U_m.reshape(U_m.shape[0],*self.N)
            for j in range(self.d):
                U_q = U_q.moveaxis(1, -1)  # move basis axis to the end for einsum
                if j==i:
                    # use second derivative matrix in the i-th dimension
                    ddPhi_j = self.bases[j].laplacian(X[j])
                    # contract along j-th dimension
                    U_q = opt_einsum.contract('B...n,Bnq->B...q', U_q, ddPhi_j)
                else:
                    # use value matrix in other dimensions
                    Phi_j = self.bases[j].forward(X[j])
                    # contract along j-th dimension
                    U_q = opt_einsum.contract('B...n,Bnq->B...q', U_q, Phi_j)
            # append gradient in direction i to gradu_q tuple
            lapu_q += (U_q,)
        #Reshape such that the last axis indexes the coordinate to which derivative has been taken
        lapu_q = torch.stack(lapu_q, axis=-1)
        return lapu_q

    def integrate(self, U_m, W_q, X_q):
        """
        Integrate the tensor-product expansion over the domain using
        provided quadrature weights and points.

        Args:
            U_m (torch.tensor): Modal coefficients with shape (B, prod(N)) or
                              (B, N_1, N_2, ..., N_d). B is batch size.
            W_q (tuple/list): Sequence of quadrature weight arrays for each dimension.
                              W_q[i] has shape (Q_i,).
            X_q (tuple/list): Sequence of quadrature point arrays for each dimension.
                              X_q[i] has shape (Q_i,).

        Returns:
            torch.tensor: Integrated values over the domain, shape (B,).
        """
        integral = U_m
        # Sequentially contract with basis evaluations and quadrature weights
        for i in range(self.d):
            Phi_i = self.bases[i].forward(X_q[i])  # (B, N_i, Q_i)
            W_i = W_q[i]  # (Q_i,)
            integral_i = opt_einsum.contract('q,Bnq->Bn', W_i, Phi_i)  # (B, N_i)
            # Combine basis evaluation and weights
            integral = opt_einsum.contract('Bn...,Bn->B...', integral, integral_i)
            # integral = integral.moveaxis(1, -1)  # move basis axis to the end for next einsum
        # After all contractions, integral has shape (B,)
        return integral
    
    def integrals(self, W_q, X_q):
        """
        Integrate the tensor-product expansion over the domain using
        provided quadrature weights and points.

        Args:
            U_m (torch.tensor): Modal coefficients with shape (B, prod(N)) or
                              (B, N_1, N_2, ..., N_d). B is batch size.
            W_q (tuple/list): Sequence of quadrature weight arrays for each dimension.
                              W_q[i] has shape (Q_i,).
            X_q (tuple/list): Sequence of quadrature point arrays for each dimension.
                              X_q[i] has shape (Q_i,).

        Returns:
            torch.tensor: Integrated values over the domain, shape (B,N_0,...,N_{d-1}).
        """
        Phi_0 = self.bases[0].forward(X_q[0])
        integrals = opt_einsum.contract('q,Bnq->Bn', W_q[0], Phi_0)  # (B, N_0)
        # Sequentially contract with basis evaluations and quadrature weights
        for i in range(1,self.d):
            Phi_i = self.bases[i].forward(X_q[i])  # (B, N_i, Q_i)
            W_i = W_q[i]  # (Q_i,)
            integral_i = opt_einsum.contract('q,Bnq->Bn', W_i, Phi_i)  # (B, N_i)
            # Combine basis evaluation and weights
            integrals = opt_einsum.contract('Bn...,Bm->Bnm...', integrals, integral_i)
            # integrals = integrals.moveaxis(1, -1)  # move basis axis to the end for next einsum
        # After all contractions, integral has shape (B,)
        return integrals