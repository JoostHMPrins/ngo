# Copyright 2025 Joost Prins

# 3rd Party
import matplotlib.pyplot as plt
import numpy as np
import opt_einsum
from scipy.interpolate import BSpline
import torch

# Local
from ngo.ml.customlayers import discretize_functions


class GaussianRBFBasis:
    """
    1D Gaussian radial-basis-function (RBF) basis.

    This class defines a set of Gaussian basis functions with fixed centers (mus)
    and shared width (sigma). It provides methods to evaluate the basis and its
    derivative at given points, and simple plotting helpers.

    Attributes:
        lambdas (float): Standard deviation (width) of the Gaussian bumps. Must be > 0.
        mus (np.ndarray): 1D array of centers of the Gaussians, shape (n,).
        n (int): Number of basis functions (len(mus)).
    """
    def __init__(self, lambdas, mus):
        self.lambdas = lambdas
        self.mus = mus
        self.n = mus.shape[0]
    
    def forward(self, x):
        """
        Evaluate the Gaussian RBFs at points x.

        Args:
            x (np.ndarray): 1D array of evaluation points, shape (n_points,).

        Returns:
            np.ndarray: Matrix of basis values, shape (n,n_points),
                        where column row is exp(-0.5 * ((x - mus[j]) / sigma)**2).
        """
        basis_values = torch.exp(-0.5 * ((x[None, None, :] - self.mus[None, :, None]) / self.lambdas[:,:,None]) ** 2)
        return basis_values
    
    def grad(self, x):
        """
        Compute derivative of each Gaussian basis function w.r.t. x.

        Args:
            x (np.ndarray): 1D array of evaluation points, shape (n_points,).

        Returns:
            np.ndarray: Matrix of basis derivatives, shape (n,n_points).
        """
        basis_gradients = -((x[None,None, None, :]  - self.mus[None,:, :, None]) / (self.lambdas[:,:,None,None] ** 2)) * self.forward(x)
        return basis_gradients
    
    def laplacian(self, x):
        """
        Compute second derivative (Laplacian) of each Gaussian basis function w.r.t. x.

        Args:
            x (np.ndarray): 1D array of evaluation points, shape (n_points,).

        Returns:
            np.ndarray: Matrix of basis second derivatives, shape (n,n_points).
        """
        basis_laplacians = (((x[None,None, None, :] - self.mus[None,:, :, None]) ** 2) / (self.lambdas[None,:,None,None] ** 4) - 1 / (self.lambdas[:,:,None,None] ** 2)) * self.forward(x)
        return basis_laplacians



class BSplineBasis1D:
    """
    A class to represent 1D B-spline basis functions.

    Attributes:
        n (int): Number of basis functions.
        p (int): Polynomial degree.
        C (int): Continuity.
        knot_vector (np.ndarray): Knot vector for the B-spline basis.
    """

    def __init__(self, n, p, C):
        """
        Initialize the B-spline basis.
        
        Args:
            n (int): Number of basis functions.
            p (int): Polynomial degree.
            C (int): Continuity.
        """
        self.n = n
        self.p = p  
        self.C = C 
        self.knot_vector = np.zeros(self.p+1)
        self.knot_vector = np.append(self.knot_vector, np.repeat(np.linspace(0, 1, int((self.n - self.p - 1)/(self.p - self.C)) + 2)[1:-1], self.p - self.C))
        self.knot_vector = np.append(self.knot_vector, np.ones(self.p+1))
    
    def forward(self, x):
        """
        Compute the B-spline basis values at given points.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Basis values at the input points. Shape: (n,n_points).
        """
        basis_values = BSpline.design_matrix(x, self.knot_vector, self.p).toarray().T
        return basis_values
    
    def grad(self, x):
        """
        Compute the gradients of the B-spline basis functions.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Gradients of the basis functions at the input points. Shape: (n,n_points).
        """
        coeffs = np.eye(self.n)
        derivative_basis_functions = 0 if self.p==0 else [BSpline(self.knot_vector, coeffs[i], self.p).derivative() for i in range(self.n)]
        basis_gradients = np.zeros((len(x),self.n)) if self.p==0 else np.vstack([dbf(x) for dbf in derivative_basis_functions])
        return basis_gradients


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
            Phi += (torch.tensor(self.bases[i].forward(X[i])),)
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
            dPhi += (torch.tensor(self.bases[i].grad(X[i])),)
        # Add batch dimension if missing
        dPhi = tuple(dPhi_i[None,:,:] if dPhi_i.ndim==2 else dPhi_i for dPhi_i in dPhi)
        return dPhi

    # def gradPhi(self, X):
    #     """
    #     Compute the gradient of the tensor-product basis functions at provided points.

    #     Args:
    #         X (tuple/list): Sequence of point arrays for each dimension.
    #                         X[i] has shape (n_points_i,).
    #     Returns:
    #         tuple: Sequence of gradient tuples for each dimension.
    #                Each gradient tuple contains basis evaluation matrices,
    #                with the derivative in the corresponding dimension.
    #     """
    #     # Compute basis evaluations and derivatives
    #     Phi = self.Phi(X)
    #     Phi = tuple(Phi_i[:,None] for Phi_i in Phi)  # add extra dim for broadcasting
    #     dPhi = self.dPhi(X)
    #     # Construct gradient tuples
    #     gradPhi = ()
    #     # For each dimension, create a tuple with dPhi in that dimension and Phi elsewhere
    #     for i in range(self.d):
    #         # Construct gradient in dimension i
    #         gradPhi_i = torch.ones_like(Phi[0])
    #         print(gradPhi_i.shape)
    #         # Multiply basis evaluations, using derivative in dimension i
    #         for j in range(self.d):
    #             # Use derivative in dimension i, value elsewhere
    #             if i==j:
    #                 print(dPhi[j].shape)
    #                 gradPhi_i *= dPhi[j]
    #             else:
    #                 print(Phi[j].shape)
    #                 gradPhi_i *= Phi[j]
    #         # Append gradient tuple for dimension i
    #         gradPhi += (gradPhi_i,)
    #     return gradPhi

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
            # start from coefficients reshaped to (N, h1, ..., hd)
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