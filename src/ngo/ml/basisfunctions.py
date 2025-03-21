# Copyright 2025 Joost Prins

# 3rd Party
import matplotlib.pyplot as plt
import numpy as np
import opt_einsum
from scipy.interpolate import BSpline

# Local
from ngo.ml.customlayers import discretize_functions

class BSplineBasis1D:
    def __init__(self, h, p, C):
        self.h = h #Number of basis functions (should equal n_el p + 1, where n_el is the number of elements)
        self.p = p  #Polynomial degree
        self.C = C #Continuity
        self.knot_vector = np.zeros(self.p+1)
        self.knot_vector = np.append(self.knot_vector, np.repeat(np.linspace(0, 1, int((self.h - self.p - 1)/(self.p - self.C)) + 2)[1:-1], self.p - self.C))
        self.knot_vector = np.append(self.knot_vector, np.ones(self.p+1))
    
    def forward(self, x):
        basis_values = BSpline.design_matrix(x, self.knot_vector, self.p).toarray()
        return basis_values
    
    def grad(self, x):
        coeffs = np.eye(self.h)
        derivative_basis_functions = [BSpline(self.knot_vector, coeffs[i], self.p).derivative() for i in range(self.h)]
        basis_gradients = np.vstack([dbf(x) for dbf in derivative_basis_functions]).T
        return basis_gradients

    def plot_1d_basis(self):
        knots = self.knot_vector
        resolution = 1000
        x_values = np.linspace(knots[self.p], knots[-self.p-1], resolution) # Adjusted range for x_values
        basis_matrix = self.forward(x_values)
        plt.figure(figsize=(8, 6))
        for i in range(self.h):
            plt.plot(x_values, basis_matrix[:,i], label=f'Basis {i}')
        plt.title(f'1D B-spline Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Basis Values')
        plt.xticks(np.array([0,1,2,3,4])/4)
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

    def plot_1d_basis(self):
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
        
    def plot_1d_basis_gradients(self):
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


class SincBasis1D:
    def __init__(self, h):
        self.h = h
        self.Dx = 1/(h-1)
        self.grid = np.linspace(0,1,self.h)
        self.tol = 1e-10

    def sinc(self, x):
        output = np.sin(x)/x
        # output[np.abs(x)<self.tol] = 1
        output[x==0] = 1
        return output
    
    def dsincdx(self, x):
        output = np.cos(x)/x - np.sin(x)/x**2
        # output[np.abs(x)<self.tol] = 0
        output[x==0] = 0
        return output
    
    def forward(self, x):
        basis_values = np.zeros((x.shape[0],self.h))
        for n in range(self.h):
            x_scaled = np.pi*(x - self.grid[n])/self.Dx
            basis_values[:, n] = self.sinc(x_scaled)
        return basis_values
    
    def grad(self, x):
        basis_gradients = np.zeros((x.shape[0],self.h))
        for n in range(self.h):
            x_scaled = np.pi*(x - self.grid[n])/self.Dx
            basis_gradients[:, n] = np.pi/self.Dx*self.dsincdx(x_scaled)
        return basis_gradients

    def plot_1d_basis(self):
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
        
    def plot_1d_basis_gradients(self):
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


class PolynomialBasis1D:
    def __init__(self, h):
        self.h = h
        self.exponents = np.arange(0,self.h)
    
    def forward(self, x):
        basis_values = x[:,None]**self.exponents[None,:]
        return basis_values
    
    def grad(self, x):
        basis_gradients = self.exponents[None,:]*x[:,None]**(self.exponents-1)
        basis_gradients[:,0] = 0
        return basis_gradients

    def plot_1d_basis(self):
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
        
    def plot_1d_basis_gradients(self):
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


class FourierBasis1D:
    def __init__(self, h):
        self.h = h
        self.n = np.arange(0,self.h)
        self.L = 1.2
        self.k_n = 2*np.pi*self.n/self.L
    
    def forward(self, x):
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
        
    def plot_1d_basis_gradients(self):
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
    def __init__(self, N_samples, variables, l_min, l_max, w, xi, N, device):
        self.basis_1d = BSplineBasis1D(h=31,p=3,C=2)
        self.bsplinebasis = TensorizedBasis([self.basis_1d,self.basis_1d])
        self.N = N
        self.w = w
        self.xi = xi
        #self.u = ManufacturedSolutionsSetDarcy(N_samples=N_samples, variables=variables, l_min=l_min, l_max=l_max, device=device).u
        #self.u_q = discretize_functions(self.u, torch.tensor(self.xi, device=device), device).detach().cpu().numpy()
        self.device = device
        # self.PODbasis = self.compute_PODbasis(self.u_q, self.N)
        # self.PODcoeffs = self.compute_PODcoeffs(self.u_q, self.N)
        self.PODcoeffs = np.load('../../../trainingdata/PODcoeffs.npy')[:self.N]
        # self.errors = self.compute_projection_errors()
        
    def compute_PODbasis(self, u_q, N):
        U, self.singularvalues, Vstar = np.linalg.svd(u_q.T, full_matrices=False)
        PODbasis = U.T
        PODbasis = PODbasis
        return PODbasis
    
    def compute_PODcoeffs(self, u_q, N):
        PODbasis_d = self.compute_PODbasis(u_q, N)
        bsplinebasis_d = self.bsplinebasis.forward(self.xi)
        d = opt_einsum.contract('q,nq,ql->nl', self.w, PODbasis_d, bsplinebasis_d)
        M = opt_einsum.contract('q,ql,qm->lm', self.w, bsplinebasis_d, bsplinebasis_d)
        M_inv = np.linalg.inv(M)
        PODcoeffs = opt_einsum.contract('lm,nl->nm',M_inv, d)
        np.save('../../../trainingdata/PODcoeffs.npy', PODcoeffs)
        return PODcoeffs
        
    def forward(self, x):
        psi = self.bsplinebasis.forward(x)
        output = opt_einsum.contract('nm,Nm->Nn', self.PODcoeffs, psi)
        return output
    
    def grad(self, x):
        gradpsi = self.bsplinebasis.grad(x)
        output = opt_einsum.contract('nm,Nmi->Nni', self.PODcoeffs, gradpsi)
        return output
    
    def compute_projection_errors(self):
        pi = self.PODbasis
        pi_proj = self.forward(self.xi).T
        L22_diff = np.sum(self.w*(pi - pi_proj)**2, axis=-1)
        L2_diff = L22_diff**(1/2)
        L22_norm = np.sum(self.w*(pi)**2, axis=-1)
        L2_norm = L22_norm**(1/2)
        L2_scaled_array = L2_diff/np.maximum(L2_norm, 1e-7*np.ones_like(L2_norm))
        return L2_scaled_array