import matplotlib.pyplot as plt
import numpy as np
import opt_einsum
from scipy.interpolate import BSpline

class BSplineBasis1D:
    def __init__(self, knot_vector, polynomial_order):
        """
        Initialize the B-spline basis generator with the given knot vectors and polynomial order.

        Args:
        knot_vectors (list of np.array): List of 1D arrays representing the knot vectors for each dimension.
        polynomial_order (int): Polynomial order (degree) of the B-splines.
        """
        self.p = polynomial_order  # Polynomial degree
        self.knot_vector = knot_vector
        self.num_basis = len(self.knot_vector) - self.p - 1

    def basis_function(self, i, p, x, knots):
        """ 
        Calculate the value of the i-th B-spline basis function of order p
        at point x, given the knot vector 'knots'.
        """
        c = np.zeros(self.num_basis)
        c[i] = 1
        bspline = BSpline(knots, c, p)
        return bspline(x)
    
    # @jit
    def forward(self, x):
        """
        Evaluate the B-spline basis functions at the given input coordinates 'x' in d dimensions.

        Args:
        x (np.array): Input coordinate array of shape (n, d) representing n points in d-dimensional space.

        Returns:
        basis_values (np.array): array of shape (n, num_basis) containing the values of all B-spline basis functions
                                     evaluated at the input coordinates 'x'.
        """
        n = x.shape[0]
        basis_values = np.zeros((n,self.num_basis))
        for i in range(self.num_basis):
            basis_values[:, i] = self.basis_function(i, self.p, x, self.knot_vector)
        return basis_values
    
    def basis_gradient(self, i, p, x, knots):
        """ 
        Calculate the analytical gradient of the i-th B-spline basis function of order p
        at point x, given the knot vector 'knots'.
        """
        c = np.zeros(self.num_basis)
        c[i] = 1
        bspline = BSpline(knots, c, p)
        bspline_derivative = bspline.derivative(1)
        return bspline_derivative(x)
    
    def grad(self, x):
        """
        Evaluate the gradients of the B-spline basis functions at the given input coordinates 'x' in d dimensions.

        Args:
        x (np.array): Input coordinate array of shape (n, d) representing n points in d-dimensional space.

        Returns:
        basis_gradients (np.array): array of shape (n, num_basis, d) containing the gradients of all B-spline
                                        basis functions evaluated at the input coordinates 'x'.
        """
        n = x.shape[0]
        basis_gradients = np.zeros((n,self.num_basis))
        for i in range(self.num_basis):
            basis_grad = self.basis_gradient(i, self.p, x, self.knot_vector)
            basis_gradients[:, i] = basis_grad
        return basis_gradients

    def plot_1d_bspline_basis(self):
        """
        Plot the 1D B-spline basis functions for a specified dimension.

        Args:
        dim_idx (int): Index of the dimension for which to plot the basis functions.
        """
        knots = self.knot_vector
        resolution = 1000
        x_values = np.linspace(knots[self.p], knots[-self.p-1], resolution)  # Adjusted range for x_values
        basis_matrix = self.forward(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.num_basis):
            plt.plot(x_values, basis_matrix[:,i], label=f'Basis {i}')
        plt.title(f'1D B-spline Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Basis Values')
        plt.legend()
        plt.grid(True)
        # plt.savefig("BSpline1D.svg", bbox_inches='tight', transparent=True)
        plt.show()
        
    def plot_1d_bspline_basis_gradients(self):
        """
        Plot the gradients of the 1D B-spline basis functions for a specified dimension.

        Args:
        dim_idx (int): Index of the dimension for which to plot the basis function gradients.
        """
        knots = self.knot_vector
        resolution = 1000
        min_knot = knots[self.p]
        max_knot = knots[-self.p-1]
        x_values = np.linspace(min_knot, max_knot, resolution)  # Use full knot span for x_values
        basis_gradients_matrix = self.grad(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.num_basis):
            plt.plot(x_values, basis_gradients_matrix[:, i], label=f'Gradient {i}')
        plt.title(f'Gradients of 1D B-spline Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Gradient Values')
        plt.legend()
        plt.grid(True)
        # plt.savefig("BSplinegrad1D.svg", bbox_inches='tight', transparent=True)
        plt.show()

# # Example usage for 1D B-spline basis functions
# knots1dx = np.array([0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1])

# polynomial_order = 3
# basis_1d = BSplineBasis1D(knots1dx, polynomial_order)

# x_values = np.linspace(0, 1, 1000)

# # Plot the 1D B-spline basis functions for dimension 0
# basis_1d.plot_1d_bspline_basis()

# # Plot the gradients of the 1D B-spline basis functions for dimension 0
# basis_1d.plot_1d_bspline_basis_gradients()


# Define the 2D B-spline basis generator
class BSplineBasis2D:
    def __init__(self, knots_x, knots_y, polynomial_order):
        self.basis_1d_x = BSplineBasis1D(knots_x, polynomial_order)
        self.basis_1d_y = BSplineBasis1D(knots_y, polynomial_order)

    def forward(self, x):
        bx = self.basis_1d_x.forward(x[:,0])
        by = self.basis_1d_y.forward(x[:,1])
        basis_values_2d = opt_einsum.contract('ni,nj->nij', bx, by)
        basis_values_2d = basis_values_2d.reshape((basis_values_2d.shape[0],basis_values_2d.shape[1]*basis_values_2d.shape[2]))
        return basis_values_2d
    
    def grad(self, x):
        bx = self.basis_1d_x.forward(x[:,0])
        by = self.basis_1d_y.forward(x[:,1])
        gx = self.basis_1d_x.grad(x[:,0])
        gy = self.basis_1d_y.grad(x[:,1])
        basis_gradients_2d_x = opt_einsum.contract('ni,nj->nij', gx, by)
        basis_gradients_2d_y = opt_einsum.contract('ni,nj->nij', bx, gy)
        basis_gradients_2d = np.zeros((basis_gradients_2d_x.shape[0],basis_gradients_2d_x.shape[1], basis_gradients_2d_x.shape[2], 2))
        basis_gradients_2d[:,:,:,0] = basis_gradients_2d_x
        basis_gradients_2d[:,:,:,1] = basis_gradients_2d_y
        basis_gradients_2d = basis_gradients_2d.reshape((basis_gradients_2d.shape[0],basis_gradients_2d.shape[1]*basis_gradients_2d.shape[2],basis_gradients_2d.shape[3]))
        return basis_gradients_2d
    
    def plot_2d_bspline_basis(self):
        resolution = 100
        x_0, x_1 = np.array(np.mgrid[0:1:resolution*1j, 0:1:resolution*1j])
        x = np.vstack([x_0.ravel(), x_1.ravel()]).T
        basis_values_2d = self.forward(x)
        num_basis_x = self.basis_1d_x.num_basis
        num_basis_y = self.basis_1d_y.num_basis
        fig, axes = plt.subplots(num_basis_x, num_basis_y, figsize=(num_basis_y*4, num_basis_x*3))
        for i in range(num_basis_x):
            for j in range(num_basis_y):
                ax = axes[i, j] if num_basis_x > 1 and num_basis_y > 1 else axes[max(i, j)]
                idx = i * num_basis_y + j
                z = basis_values_2d[:, idx].reshape(resolution, resolution)
                c = ax.contourf(x_0, x_1, z, levels=20)
                fig.colorbar(c, ax=ax)
                ax.set_title(f'Basis ({i},{j})')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
        plt.tight_layout()
        # plt.savefig("BSpline2D.svg", bbox_inches='tight', transparent=True)
        plt.show()
        
    def plot_2d_bspline_basis_gradients(self):
        resolution = 100
        x_0, x_1 = np.array(np.mgrid[0:1:resolution*1j, 0:1:resolution*1j])
        x = np.vstack([x_0.ravel(), x_1.ravel()]).T
        basis_gradients_2d = self.grad(x)
        num_basis_x = self.basis_1d_x.num_basis
        num_basis_y = self.basis_1d_y.num_basis
        fig, axes = plt.subplots(num_basis_x, num_basis_y, figsize=(num_basis_y*4, num_basis_x*3))
        for i in range(num_basis_x):
            for j in range(num_basis_y):
                ax = axes[i, j] if num_basis_x > 1 and num_basis_y > 1 else axes[max(i, j)]
                idx = i * num_basis_y + j
                z = np.linalg.norm(basis_gradients_2d[:, idx], axis=-1).reshape(resolution, resolution)
                c = ax.contourf(x_0, x_1, z, levels=20)
                fig.colorbar(c, ax=ax)
                ax.set_title(f'Gradient ({i},{j})')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
        plt.tight_layout()
        # plt.savefig("BSplinegrad2D.svg", bbox_inches='tight', transparent=True)
        plt.show()

# # Example usage of the 2D B-spline basis generator
# knots1dx = np.array([0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1])
# knots1dy = np.array([0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1])
# polynomial_order = 3

# # Create the 2D B-spline basis generator
# basis_2d = BSplineBasis2D(knots1dx, knots1dy, polynomial_order)

# # Plot the 2D B-spline basis functions
# basis_2d.plot_2d_bspline_basis()
# basis_2d.plot_2d_bspline_basis_gradients()


class BSplineInterpolator2D:
    def __init__(self, x_data, u_data, knots_x, knots_y, polynomial_order):
        self.basis_2d = BSplineBasis2D(knots_x, knots_y, polynomial_order)
        self.coeffs = self.compute_coeffs(x_data, u_data)
        
    def compute_X(self, x_data):
        return self.basis_2d.forward(x_data)
    
    def compute_coeffs(self, x_data, u_data):
        X = self.compute_X(x_data)
        coeffs = np.linalg.lstsq(X, u_data)[0]
        return coeffs
        
    def forward(self, x):
        psi = self.basis_2d.forward(x)
        output = opt_einsum.contract('n,Nn->N', self.coeffs, psi)
        return output
    
    def grad(self, x):
        gradpsi = self.basis_2d.grad(x)
        output = opt_einsum.contract('n,Nni->Ni', self.coeffs, gradpsi)
        return output