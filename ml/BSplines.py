import torch
import matplotlib.pyplot as plt
import numpy as np

class BSplineBasis1D:
    def __init__(self, knot_vector, polynomial_order, **kwargs):
        """
        Initialize the B-spline basis generator with the given knot vectors and polynomial order.

        Args:
        knot_vectors (list of torch.Tensor): List of 1D tensors representing the knot vectors for each dimension.
        polynomial_order (int): Polynomial order (degree) of the B-splines.
        """
        self.p = polynomial_order  # Polynomial degree
        self.knot_vector = knot_vector
        self.num_basis = len(self.knot_vector) - self.p - 1
        self.dtype = kwargs['hparams']['dtype']

    def basis_function(self, i, p, x, knots):
        """ 
        Calculate the value of the i-th B-spline basis function of order p
        at point x, given the knot vector 'knots'.
        """
        if p == 0:
            return torch.where((knots[i] <= x) & (x <= knots[i+1]), torch.tensor(1.0), torch.tensor(0.0))
        
        denom1 = knots[i+p] - knots[i]
        term1 = torch.tensor(0.0) if denom1 == 0.0 else ((x - knots[i]) / denom1) * self.basis_function(i, p-1, x, knots)
        
        denom2 = knots[i+p+1] - knots[i+1]
        term2 = torch.tensor(0.0) if denom2 == 0.0 else ((knots[i+p+1] - x) / denom2) * self.basis_function(i+1, p-1, x, knots)
        
        return term1 + term2
    
    def forward(self, x):
        """
        Evaluate the B-spline basis functions at the given input coordinates 'x' in d dimensions.

        Args:
        x (torch.Tensor): Input coordinate tensor of shape (n, d) representing n points in d-dimensional space.

        Returns:
        basis_values (torch.Tensor): Tensor of shape (n, num_basis) containing the values of all B-spline basis functions
                                     evaluated at the input coordinates 'x'.
        """
        n = x.shape[0]
        basis_values = torch.zeros(n, self.num_basis, dtype=self.dtype)

        idx = 0
        knots = self.knot_vector
            
        for i in range(self.num_basis):
            basis_values[:, i] = self.basis_function(i, self.p, x, knots)
        
        return basis_values

    def basis_gradient(self, i, p, x, knots):
        """ 
        Calculate the analytical gradient of the i-th B-spline basis function of order p
        at point x, given the knot vector 'knots'.
        """
        if p == 0:
            return torch.tensor(0.0)
        
        basis_p_minus_1_left = self.basis_function(i, p-1, x, knots)
        basis_p_minus_1_right = self.basis_function(i+1, p-1, x, knots)
        
        denom1 = knots[i+p] - knots[i]
        denom2 = knots[i+p+1] - knots[i+1]
        
        gradient_left = (p / denom1) * basis_p_minus_1_left if denom1 != 0.0 else torch.tensor(0.0)
        gradient_right = -(p / denom2) * basis_p_minus_1_right if denom2 != 0.0 else torch.tensor(0.0)
        
        return gradient_left + gradient_right
    
    def grad(self, x):
        """
        Evaluate the gradients of the B-spline basis functions at the given input coordinates 'x' in d dimensions.

        Args:
        x (torch.Tensor): Input coordinate tensor of shape (n, d) representing n points in d-dimensional space.

        Returns:
        basis_gradients (torch.Tensor): Tensor of shape (n, num_basis, d) containing the gradients of all B-spline
                                        basis functions evaluated at the input coordinates 'x'.
        """
        n = x.shape[0]
        basis_gradients = torch.zeros(n, self.num_basis, dtype=self.dtype)

        knots = self.knot_vector

        for i in range(self.num_basis):
            basis_grad = self.basis_gradient(i, self.p, x, knots)
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
        x_values = torch.linspace(knots[self.p], knots[-self.p-1], resolution)  # Adjusted range for x_values
        basis_matrix = self.forward(x_values)

        plt.figure(figsize=(8, 6))
        for i in range(self.num_basis):
            plt.plot(x_values.numpy(), basis_matrix[:,i].numpy(), label=f'Basis {i}')

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
        x_values = torch.linspace(min_knot, max_knot, resolution)  # Use full knot span for x_values
        basis_gradients_matrix = self.grad(x_values)

        plt.figure(figsize=(8, 6))
        for i in range(self.num_basis):
            plt.plot(x_values.numpy(), basis_gradients_matrix[:, i].numpy(), label=f'Gradient {i}')

        plt.title(f'Gradients of 1D B-spline Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Gradient Values')
        plt.legend()
        plt.grid(True)
        # plt.savefig("BSplinegrad1D.svg", bbox_inches='tight', transparent=True)
        plt.show()

# # Example usage for 1D B-spline basis functions
# knots1dx = torch.tensor([0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1])

# polynomial_order = 3
# basis_1d = BSplineBasis1D(knots1dx, polynomial_order)

# x_values = torch.linspace(0, 1, 1000)

# # Plot the 1D B-spline basis functions for dimension 0
# basis_1d.plot_1d_bspline_basis()

# # Plot the gradients of the 1D B-spline basis functions for dimension 0
# basis_1d.plot_1d_bspline_basis_gradients()


# Define the 2D B-spline basis generator
class BSplineBasis2D:
    def __init__(self, knots_x, knots_y, polynomial_order, **kwargs):
        self.basis_1d_x = BSplineBasis1D(knots_x, polynomial_order, **kwargs)
        self.basis_1d_y = BSplineBasis1D(knots_y, polynomial_order, **kwargs)
        self.dtype = kwargs['hparams']['dtype']

    def forward(self, x):
        bx = self.basis_1d_x.forward(x[:,0])
        by = self.basis_1d_y.forward(x[:,1])
        basis_values_2d = torch.einsum('ni,nj->nij', bx, by).flatten(-2,-1)
        return basis_values_2d
    
    def grad(self, x):
        bx = self.basis_1d_x.forward(x[:,0])
        by = self.basis_1d_y.forward(x[:,1])
        gx = self.basis_1d_x.grad(x[:,0])
        gy = self.basis_1d_y.grad(x[:,1])
        basis_gradients_2d_x = torch.einsum('ni,nj->nij', gx, by)
        basis_gradients_2d_y = torch.einsum('ni,nj->nij', bx, gy)
        basis_gradients_2d = torch.zeros((basis_gradients_2d_x.shape[0],basis_gradients_2d_x.shape[1], basis_gradients_2d_x.shape[2], 2), dtype=self.dtype)
        basis_gradients_2d[:,:,:,0] = basis_gradients_2d_x
        basis_gradients_2d[:,:,:,1] = basis_gradients_2d_y
        basis_gradients_2d = basis_gradients_2d.flatten(-3,-2)
        return basis_gradients_2d
    
    def plot_2d_bspline_basis(self):
        resolution = 100
        x_0, x_1 = torch.tensor(np.mgrid[0:1:resolution*1j, 0:1:resolution*1j])
        x = torch.vstack([x_0.ravel(), x_1.ravel()]).T
        basis_values_2d = self.forward(x)
        
        num_basis_x = self.basis_1d_x.num_basis
        num_basis_y = self.basis_1d_y.num_basis
        
        fig, axes = plt.subplots(num_basis_x, num_basis_y, figsize=(num_basis_y*4, num_basis_x*3))
        
        for i in range(num_basis_x):
            for j in range(num_basis_y):
                ax = axes[i, j] if num_basis_x > 1 and num_basis_y > 1 else axes[max(i, j)]
                idx = i * num_basis_y + j
                z = basis_values_2d[:, idx].reshape(resolution, resolution)
                c = ax.contourf(x_0.numpy(), x_1.numpy(), z.numpy(), levels=20)
                fig.colorbar(c, ax=ax)
                ax.set_title(f'Basis ({i},{j})')
                ax.set_xlabel('x')
                ax.set_ylabel('y')

        plt.tight_layout()
        # plt.savefig("BSpline2D.svg", bbox_inches='tight', transparent=True)
        plt.show()
        
    def plot_2d_bspline_basis_gradients(self):
        resolution = 100
        x_0, x_1 = torch.tensor(np.mgrid[0:1:resolution*1j, 0:1:resolution*1j])
        x = torch.vstack([x_0.ravel(), x_1.ravel()]).T
        basis_gradients_2d = self.grad(x)

        num_basis_x = self.basis_1d_x.num_basis
        num_basis_y = self.basis_1d_y.num_basis

        fig, axes = plt.subplots(num_basis_x, num_basis_y, figsize=(num_basis_y*4, num_basis_x*3))

        for i in range(num_basis_x):
            for j in range(num_basis_y):
                ax = axes[i, j] if num_basis_x > 1 and num_basis_y > 1 else axes[max(i, j)]
                idx = i * num_basis_y + j
                z = torch.norm(basis_gradients_2d[:, idx], dim=-1).reshape(resolution, resolution)
                c = ax.contourf(x_0.numpy(), x_1.numpy(), z.numpy(), levels=20)
                fig.colorbar(c, ax=ax)
                ax.set_title(f'Gradient ({i},{j})')
                ax.set_xlabel('x')
                ax.set_ylabel('y')

        plt.tight_layout()
        # plt.savefig("BSplinegrad2D.svg", bbox_inches='tight', transparent=True)
        plt.show()


# # Example usage of the 2D B-spline basis generator
# knots1dx = torch.tensor([0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1])
# knots1dy = torch.tensor([0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1])
# polynomial_order = 3

# # Create the 2D B-spline basis generator
# basis_2d = BSplineBasis2D(knots1dx, knots1dy, polynomial_order)

# # Plot the 2D B-spline basis functions
# basis_2d.plot_2d_bspline_basis()
# basis_2d.plot_2d_bspline_basis_gradients()