# Copyright 2025 Joost Prins

# 3rd Party
import matplotlib.pyplot as plt
import numpy as np
import opt_einsum
from scipy.interpolate import BSpline

# Local
from ngo.ml.customlayers import discretize_functions


class BSplineBasis1D:
    """
    A class to represent 1D B-spline basis functions.

    Attributes:
        h (int): Number of basis functions.
        p (int): Polynomial degree.
        C (int): Continuity.
        knot_vector (np.ndarray): Knot vector for the B-spline basis.
    """

    def __init__(self, h, p, C):
        """
        Initialize the B-spline basis.
        
        Args:
            h (int): Number of basis functions.
            p (int): Polynomial degree.
            C (int): Continuity.
        """
        self.h = h
        self.p = p  
        self.C = C 
        self.knot_vector = np.zeros(self.p+1)
        self.knot_vector = np.append(self.knot_vector, np.repeat(np.linspace(0, 1, int((self.h - self.p - 1)/(self.p - self.C)) + 2)[1:-1], self.p - self.C))
        self.knot_vector = np.append(self.knot_vector, np.ones(self.p+1))
    
    def forward(self, x):
        """
        Compute the B-spline basis values at given points.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Basis values at the input points. Shape: (n_points, h).
        """
        basis_values = BSpline.design_matrix(x, self.knot_vector, self.p).toarray()
        return basis_values
    
    def grad(self, x):
        """
        Compute the gradients of the B-spline basis functions.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Gradients of the basis functions at the input points. Shape: (n_points, h).
        """
        coeffs = np.eye(self.h)
        derivative_basis_functions = 0 if self.p==0 else [BSpline(self.knot_vector, coeffs[i], self.p).derivative() for i in range(self.h)]
        basis_gradients = np.zeros((len(x),self.h)) if self.p==0 else np.vstack([dbf(x) for dbf in derivative_basis_functions]).T
        return basis_gradients

    def plot_1d_basis(self):
        """
        Plot the 1D B-spline basis functions.
        """
        knots = self.knot_vector
        resolution = 1000
        x_values = np.linspace(knots[self.p], knots[-self.p-1], resolution) # Adjusted range for x_values
        basis_matrix = self.forward(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.h):
            plt.plot(x_values, basis_matrix[:,i], label=f'Basis {i}')
        plt.title(f'1D Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Basis Values')
        plt.xticks(np.array([0,1,2,3,4])/4)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_1d_basis_gradients(self):
        """
        Plot the gradients of the 1D B-spline basis functions.
        """
        knots = self.knot_vector
        resolution = 1000
        min_knot = knots[self.p]
        max_knot = knots[-self.p-1]
        x_values = np.linspace(min_knot, max_knot, resolution)  # Use full knot span for x_values
        basis_gradients_matrix = self.grad(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.h):
            plt.plot(x_values, basis_gradients_matrix[:, i], label=f'Gradient {i}')
        plt.title(f'Gradients of 1D Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Gradient Values')
        plt.legend()
        plt.grid(True)
        plt.show()


class ChebyshevTBasis1D:
    """
    A class to represent 1D Chebyshev polynomial basis functions of the first kind.

    Attributes:
        h (int): Number of basis functions.
    """
    def __init__(self, h):
        """
        Initialize the Chebyshev basis.

        Args:
            h (int): Number of basis functions.
        """
        self.h = h

    def basis_function(self, n):
        """
        Compute the nth Chebyshev basis function.

        Args:
            n (int): Index of the basis function.

        Returns:
            np.polynomial.chebyshev.Chebyshev: The nth Chebyshev basis function.
        """
        c = np.zeros(self.h)
        c[n] = 1
        basisfunction = np.polynomial.chebyshev.Chebyshev(coef=c,domain=[0,1])
        return basisfunction
    
    def forward(self, x):
        """
        Compute the Chebyshev basis values at given points.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Basis values at the input points. Shape: (n_points, h).
        """
        basis_values = np.zeros((x.shape[0],self.h))
        for n in range(self.h):
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
            np.ndarray: Gradients of the basis functions at the input points. Shape: (n_points, h).
        """
        basis_gradients = np.zeros((x.shape[0],self.h))
        for n in range(self.h):
            basis_grad = self.basis_gradient(n)(x)
            basis_gradients[:, n] = basis_grad
        return basis_gradients

    def plot_1d_basis(self):
        """
        Plot the 1D Chebyshev basis functions.
        """
        resolution = 1000
        x_values = np.linspace(0, 1, resolution)  # Adjusted range for x_values
        basis_matrix = self.forward(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.h):
            plt.plot(x_values, basis_matrix[:,i], label=f'Basis {i}')
        plt.title(f'1D Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Basis Values')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_1d_basis_gradients(self):
        """
        Plot the gradients of the 1D Chebyshev basis functions.
        """
        resolution = 1000
        x_values = np.linspace(0, 1, resolution)  # Use full knot span for x_values
        basis_gradients_matrix = self.grad(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.h):
            plt.plot(x_values, basis_gradients_matrix[:, i], label=f'Gradient {i}')
        plt.title(f'Gradients of 1D Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Gradient Values')
        plt.legend()
        plt.grid(True)
        plt.show()


class SincBasis1D:
    """
    A class to represent 1D sinc basis functions (sin(x)/x).

    Attributes:
        h (int): Number of basis functions.
        Dx (float): Spacing between grid points.
        grid (np.ndarray): Grid points for the basis functions. Shape: (h,).
    """

    def __init__(self, h):
        """
        Initialize the sinc basis.

        Args:
            h (int): Number of basis functions.
        """
        self.h = h
        self.Dx = 1/(h-1)
        self.grid = np.linspace(0,1,self.h)

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
            np.ndarray: Basis values at the input points. Shape: (n_points, h).
        """
        basis_values = np.zeros((x.shape[0],self.h))
        for n in range(self.h):
            x_scaled = np.pi*(x - self.grid[n])/self.Dx
            basis_values[:, n] = self.sinc(x_scaled)
        return basis_values
    
    def grad(self, x):
        """
        Compute the gradients of the sinc basis functions.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Gradients of the basis functions at the input points. Shape: (n_points, h).
        """
        basis_gradients = np.zeros((x.shape[0],self.h))
        for n in range(self.h):
            x_scaled = np.pi*(x - self.grid[n])/self.Dx
            basis_gradients[:, n] = np.pi/self.Dx*self.dsincdx(x_scaled)
        return basis_gradients

    def plot_1d_basis(self):
        """
        Plot the 1D sinc basis functions.
        """
        resolution = 1000
        x_values = np.linspace(0, 1, resolution)  # Adjusted range for x_values
        basis_matrix = self.forward(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.h):
            plt.plot(x_values, basis_matrix[:,i], label=f'Basis {i}')
        plt.title(f'1D Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Basis Values')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_1d_basis_gradients(self):
        """
        Plot the gradients of the 1D sinc basis functions.
        """
        resolution = 1000
        x_values = np.linspace(0, 1, resolution)  # Use full knot span for x_values
        basis_gradients_matrix = self.grad(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.h):
            plt.plot(x_values, basis_gradients_matrix[:, i], label=f'Gradient {i}')
        plt.title(f'Gradients of 1D Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Gradient Values')
        plt.legend()
        plt.grid(True)
        plt.show()


class PolynomialBasis1D:
    """
    A class to represent 1D polynomial basis functions.
    The basis functions are defined as:
    phi_0 = c_0, phi_1 = c_1 * x, phi_2 = c_2 * x^2, ..., phi_h = c_h * x^h.

    Attributes:
        h (int): Number of basis functions.
        exponents (np.ndarray): Array of the exponents of the individual basis functions.
    """
    def __init__(self, h):
        """
        Initialize the polynomial basis.

        Args:
            h (int): Number of basis functions.
        """
        self.h = h
        self.exponents = np.arange(0,self.h)
    
    def forward(self, x):
        """
        Compute the polynomial basis values at given points.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Basis values at the input points. Shape: (n_points, h).
        """
        basis_values = x[:,None]**self.exponents[None,:]
        return basis_values
    
    def grad(self, x):
        """
        Compute the gradients of the polynomial basis functions.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Gradients of the basis functions at the input points. Shape: (n_points, h).
        """
        basis_gradients = self.exponents[None,:]*x[:,None]**(self.exponents-1)
        basis_gradients[:,0] = 0
        return basis_gradients

    def plot_1d_basis(self):
        """
        Plot the 1D polynomial basis functions.
        """
        resolution = 1000
        x_values = np.linspace(0, 1, resolution)  # Adjusted range for x_values
        basis_matrix = self.forward(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.h):
            plt.plot(x_values, basis_matrix[:,i], label=f'Basis {i}')
        plt.title(f'1D Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Basis Values')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_1d_basis_gradients(self):
        """
        Plot the gradients of the 1D polynomial basis functions.
        """
        resolution = 1000
        x_values = np.linspace(0, 1, resolution)  # Use full knot span for x_values
        basis_gradients_matrix = self.grad(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.h):
            plt.plot(x_values, basis_gradients_matrix[:, i], label=f'Gradient {i}')
        plt.title(f'Gradients of 1D Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Gradient Values')
        plt.legend()
        plt.grid(True)
        plt.show()


class FourierBasis1D:
    """
    A class to represent 1D Fourier basis functions in sine-cosine form.
    The Fourier basis functions are defined as:
    phi_0 = 1, phi_1 = cos(k_1 * x), phi_2 = sin(k_1 * x), ..., 
    alternating between cosine and sine terms.

    Attributes:
        h (int): Number of basis functions.
        n (np.ndarray): Array of indices for the basis functions.
        L (float): Length of the domain (should be unequal to 1 to allow for aperiodicity).
        k_n (np.ndarray): Wave numbers for the basis functions.
    """
    def __init__(self, h):
        """
        Initialize the Fourier basis.

        Args:
            h (int): Number of basis functions.
        """
        self.h = h
        self.n = np.arange(0,self.h)
        self.L = 1.2
        self.k_n = 2*np.pi*self.n/self.L
    
    def forward(self, x):
        """
        Compute the Fourier basis values at given points.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Basis values at the input points. Shape: (n_points, h).
        """
        basis_values = np.zeros((x.shape[0],self.h))
        n = 0
        for i in np.arange(0,self.h,2):
            basis_values[:,i] = np.cos(self.k_n[n]*x)
            n+=1
        n = 1
        for i in np.arange(1,self.h,2):
            basis_values[:,i] = np.sin(self.k_n[n]*x)
            n+=1
        return basis_values
    
    def grad(self, x):
        """
        Compute the gradients of the Fourier basis functions.

        Args:
            x (np.ndarray): Input points. Shape: (n_points,).

        Returns:
            np.ndarray: Gradients of the basis functions at the input points. Shape: (n_points, h).
        """
        basis_gradients = np.zeros((x.shape[0],self.h))
        n = 0
        for i in np.arange(0,self.h,2):
            basis_gradients[:,i] = -self.k_n[n]*np.sin(self.k_n[n]*x)
            n+=1
        n = 1
        for i in np.arange(1,self.h,2):
            basis_gradients[:,i] = self.k_n[n]*np.cos(self.k_n[n]*x)
            n+=1
        return basis_gradients

    def plot_1d_basis(self):
        """
        Plot the 1D Fourier basis functions.
        """
        resolution = 1000
        x_values = np.linspace(0, 1, resolution)  # Adjusted range for x_values
        basis_matrix = self.forward(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.h):
            plt.plot(x_values, basis_matrix[:,i], label=f'Basis {i}')
        plt.title(f'1D Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Basis Values')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_1d_basis_gradients(self):
        """
        Plot the gradients of the 1D Fourier basis functions.
        """
        resolution = 1000
        x_values = np.linspace(0, 1, resolution)  # Use full knot span for x_values
        basis_gradients_matrix = self.grad(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.h):
            plt.plot(x_values, basis_gradients_matrix[:, i], label=f'Gradient {i}')
        plt.title(f'Gradients of 1D Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Gradient Values')
        plt.legend()
        plt.grid(True)
        plt.show()


class TensorizedBasis:
    """
    A class to represent tensorized basis functions.
    This class combines multiple 1D basis functions into a higher-dimensional tensorized/Kronecker product basis.

    Attributes:
        bases (list): A list of 1D basis objects.
        d (int): Dimensionality of the tensorized basis (number of 1D bases).
        n_basisfunctions (int): Total number of basis functions in the tensorized basis.
    """
    def __init__(self, bases):
        """
        Initialize the tensorized basis.

        Args:
            bases (list): A list of 1D basis objects.
        """
        self.bases = bases
        self.d = len(bases)
        self.n_basisfunctions = 1
        for i in range(len(bases)):
            self.n_basisfunctions *= bases[i].h
    
    def forward(self, x):
        """
        Compute the tensorized basis values at given points.

        Args:
            x (np.ndarray): Input points. Shape: (n_points, d).

        Returns:
            np.ndarray: Tensorized basis values at the input points. Shape: (n_points, n_basisfunctions).
        """
        tensorproduct = self.bases[0].forward(x[:,0])
        newsize = self.bases[0].h
        for i in range(1,self.d):
            newsize = newsize*self.bases[i].h
            tensorproduct = opt_einsum.contract('Nm,Nn->Nmn', tensorproduct, self.bases[i].forward(x[:,i])).reshape(x.shape[0], newsize)
        return tensorproduct
    
    def grad(self, x):
        """
        Compute the gradients of the tensorized basis functions.

        Args:
            x (np.ndarray): Input points. Shape: (n_points, d).
            
        Returns:
            np.ndarray: Gradients of the tensorized basis functions at the input points.
                        Shape: (n_points, n_basisfunctions, d).
        """
        tensorproduct = np.zeros((x.shape[0], self.n_basisfunctions, self.d))
        for i in range(self.d):
            if i==0:
                tensorproduct_i = self.bases[0].grad(x[:,0])
            else:
                tensorproduct_i = self.bases[0].forward(x[:,0])
            newsize = self.bases[0].h
            for j in range(1,self.d):
                newsize = newsize*self.bases[j].h
                if j==i:
                    tensorproduct_i = opt_einsum.contract('Nm,Nn->Nmn', tensorproduct_i, self.bases[j].grad(x[:,j])).reshape(x.shape[0], newsize)
                else: 
                    tensorproduct_i = opt_einsum.contract('Nm,Nn->Nmn', tensorproduct_i, self.bases[j].forward(x[:,j])).reshape(x.shape[0], newsize)
            tensorproduct[:,:,i] = tensorproduct_i
        return tensorproduct
    
    def plot_2d_basis(self):
        """
        Plot the 2D tensorized basis functions.
        This method assumes the tensorized basis is 2D (d=2).
        """
        resolution = 100
        x_0, x_1 = np.mgrid[0:1:resolution*1j, 0:1:resolution*1j]
        x = np.vstack([x_0.ravel(), x_1.ravel()]).T
        basis_values_2d = self.forward(x)
        h_x = self.bases[0].h
        h_y = self.bases[1].h
        fig, axes = plt.subplots(h_x, h_y, figsize=(h_y*4, h_x*3))
        for i in range(h_x):
            for j in range(h_y):
                ax = axes[i, j] if h_x > 1 and h_y > 1 else axes[max(i, j)]
                idx = i * h_y + j
                z = basis_values_2d[:, idx]
                c = ax.tripcolor(x[:,0], x[:,1], z)
                fig.colorbar(c, ax=ax)
                ax.set_title(f'Basis ({i},{j})')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
        plt.tight_layout()
        plt.show()
        
    def plot_2d_basis_gradients(self):
        """
        Plot the gradients of the 2D tensorized basis functions.
        This method assumes the tensorized basis is 2D (d=2).
        """
        resolution = 100
        x_0, x_1 = np.mgrid[0:1:resolution*1j, 0:1:resolution*1j]
        x = np.vstack([x_0.ravel(), x_1.ravel()]).T
        basis_gradients_2d = self.grad(x)
        h_x = self.bases[0].h
        h_y = self.bases[1].h
        fig, axes = plt.subplots(h_x, h_y, figsize=(h_y*4, h_x*3))
        for i in range(h_x):
            for j in range(h_y):
                ax = axes[i, j] if h_x > 1 and h_y > 1 else axes[max(i, j)]
                idx = i * h_y + j
                z = np.sum(basis_gradients_2d[:, idx, :]**2, axis=-1)**(1/2)
                c = ax.tripcolor(x[:,0], x[:,1], z)
                fig.colorbar(c, ax=ax)
                ax.set_title(f'Gradient ({i},{j})')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
        plt.tight_layout()
        plt.show()