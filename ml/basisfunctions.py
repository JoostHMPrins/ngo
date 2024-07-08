import matplotlib.pyplot as plt
import numpy as np
import opt_einsum
from scipy.interpolate import BSpline

class BSplineBasis1D:
    def __init__(self, h, p, C):
        self.h = h #Number of basis functions (should equal n_el p + 1, where n_el is the number of elements)
        self.p = p  #Polynomial degree
        self.C = C #Continuity
        self.knot_vector = np.zeros(self.p+1)
        self.knot_vector = np.append(self.knot_vector, np.repeat(np.linspace(0, 1, int((self.h - self.p - 1)/(self.p - self.C)) + 2)[1:-1], self.p - self.C))
        self.knot_vector = np.append(self.knot_vector, np.ones(self.p+1))

    def basis_function(self, n, p, x, knots):
        c = np.zeros(self.h)
        c[n] = 1
        bspline = BSpline(knots, c, p)
        return bspline(x)
    
    def forward(self, x):
        basis_values = np.zeros((x.shape[0],self.h))
        for n in range(self.h):
            basis_values[:, n] = self.basis_function(n, self.p, x, self.knot_vector)
        return basis_values
    
    def basis_gradient(self, n, p, x, knots):
        c = np.zeros(self.h)
        c[n] = 1
        bspline = BSpline(knots, c, p)
        bspline_derivative = bspline.derivative(1)
        return bspline_derivative(x)
    
    def grad(self, x):
        basis_gradients = np.zeros((x.shape[0],self.h))
        for n in range(self.h):
            basis_grad = self.basis_gradient(n, self.p, x, self.knot_vector)
            basis_gradients[:, n] = basis_grad
        return basis_gradients

    def plot_1d_basis(self):
        knots = self.knot_vector
        resolution = 1000
        x_values = np.linspace(knots[self.p], knots[-self.p-1], resolution)  # Adjusted range for x_values
        basis_matrix = self.forward(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.h):
            plt.plot(x_values, basis_matrix[:,i], label=f'Basis {i}')
        plt.title(f'1D B-spline Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Basis Values')
        plt.legend()
        plt.grid(True)
        # plt.savefig("BSpline1D.svg", bbox_inches='tight', transparent=True)
        plt.show()
        
    def plot_1d_basis_gradients(self):
        knots = self.knot_vector
        resolution = 1000
        min_knot = knots[self.p]
        max_knot = knots[-self.p-1]
        x_values = np.linspace(min_knot, max_knot, resolution)  # Use full knot span for x_values
        basis_gradients_matrix = self.grad(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.h):
            plt.plot(x_values, basis_gradients_matrix[:, i], label=f'Gradient {i}')
        plt.title(f'Gradients of 1D B-spline Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Gradient Values')
        plt.legend()
        plt.grid(True)
        # plt.savefig("BSplinegrad1D.svg", bbox_inches='tight', transparent=True)
        plt.show()


class ChebyshevTBasis1D:
    def __init__(self, h):
        self.h = h # Number of basis functions

    def basis_function(self, n):
        c = np.zeros(self.h)
        c[n] = 1

        basisfunction = np.polynomial.chebyshev.Chebyshev(coef=c,domain=[0,1])
        return basisfunction
    
    def forward(self, x):
        basis_values = np.zeros((x.shape[0],self.h))
        for n in range(self.h):
            basis_values[:, n] = self.basis_function(n)(x)
        return basis_values
    
    def basis_gradient(self, n):
        basisfunction = self.basis_function(n)
        derivative = basisfunction.deriv(m=1)
        return derivative
    
    def grad(self, x):
        basis_gradients = np.zeros((x.shape[0],self.h))
        for n in range(self.h):
            basis_grad = self.basis_gradient(n)(x)
            basis_gradients[:, n] = basis_grad
        return basis_gradients

    def plot_1d_bspline_basis(self):
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
        # plt.savefig("BSpline1D.svg", bbox_inches='tight', transparent=True)
        plt.show()
        
    def plot_1d_bspline_basis_gradients(self):
        """
        Plot the gradients of the 1D B-spline basis functions for a specified dimension.

        Args:
        dim_idx (int): Index of the dimension for which to plot the basis function gradients.
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
        # plt.savefig("BSplinegrad1D.svg", bbox_inches='tight', transparent=True)
        plt.show()

# Example usage for 1D B-spline basis functions
# basis_1d = ChebyshevTBasis1D(h=8)

# # Plot the 1D B-spline basis functions for dimension 0
# basis_1d.plot_1d_bspline_basis()

# # Plot the gradients of the 1D B-spline basis functions for dimension 0
# basis_1d.plot_1d_bspline_basis_gradients()


class TensorizedBasis:
    def __init__(self, bases):
        self.bases = bases
        self.d = len(bases)
        self.n_basisfunctions = 1
        for i in range(len(bases)):
            self.n_basisfunctions *= bases[i].h
    
    def forward(self, x):
        tensorproduct = self.bases[0].forward(x[:,0])
        newsize = self.bases[0].h
        for i in range(1,self.d):
            newsize = newsize*self.bases[i].h
            tensorproduct = opt_einsum.contract('Nm,Nn->Nmn', tensorproduct, self.bases[i].forward(x[:,i])).reshape(x.shape[0], newsize)
        return tensorproduct
    
    def grad(self, x):
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
        # plt.savefig("BSpline2D.svg", bbox_inches='tight', transparent=True)
        plt.show()
        
    def plot_2d_basis_gradients(self):
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
                # z = basis_gradients_2d[:, idx]
                c = ax.tripcolor(x[:,0], x[:,1], z)
                fig.colorbar(c, ax=ax)
                ax.set_title(f'Gradient ({i},{j})')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
        plt.tight_layout()
        # plt.savefig("BSplinegrad2D.svg", bbox_inches='tight', transparent=True)
        plt.show()


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


class BSplineInterpolatedPOD2D:
    def __init__(self, x_data, u_data, h, knots_x, knots_y, polynomial_order):
        self.basis_2d = BSplineBasis2D(knots_x, knots_y, polynomial_order)
        self.h = self.basis_2d.basis_1d_x.h
        self.p = self.basis_2d.p
        # self.PODbasis = self.compute_PODbasis(u_data, h)
        # self.PODcoeffs = self.compute_PODcoeffs(x_data, u_data, h, knots_x, knots_y, polynomial_order)
        self.PODcoeffs = np.load('../../../trainingdata/PODcoeffs.npy')
        # self.errors = self.compute_projection_errors(x_data)
        
    def compute_PODbasis(self, u_data, h):
        U, self.singularvalues, Vstar = np.linalg.svd(u_data.T, full_matrices=False)
        PODbasis = U.T
        PODbasis_truncated = PODbasis[:h]
        return PODbasis_truncated
    
    def compute_PODcoeffs(self, x_data, u_data, h, knots_x, knots_y, polynomial_order):
        PODbasis = self.compute_PODbasis(u_data, h)
        PODcoeffs = []
        for i in range(len(PODbasis)):
            interpolator = BSplineInterpolator2D(x_data, PODbasis[i], knots_x, knots_y, polynomial_order)
            PODcoeffs.append(interpolator.compute_coeffs(x_data, PODbasis[i]))
        PODcoeffs = np.array(PODcoeffs)
        np.save('../../../PODcoeffs.npy', PODcoeffs)
        return PODcoeffs
        
    def forward(self, x):
        psi = self.basis_2d.forward(x)
        output = opt_einsum.contract('nm,Nm->Nn', self.PODcoeffs, psi)
        return output
    
    def grad(self, x):
        gradpsi = self.basis_2d.grad(x)
        output = opt_einsum.contract('nm,Nmi->Nni', self.PODcoeffs, gradpsi)
        return output
    
    def compute_projection_errors(self, x_data):
        PODbasis_int = self.forward(x_data)
        errors = np.linalg.norm(PODbasis_int.T - self.PODbasis, ord=2, axis=-1)/np.linalg.norm(self.PODbasis, ord=2, axis=-1)
        return errors