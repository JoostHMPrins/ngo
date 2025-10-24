import numpy as np
from scipy.spatial import distance_matrix
import opt_einsum
# 
import jax
import jax.numpy as jnp
from jax import jit, vmap

class GRF:
    def __init__(self, N_samples, l):
        super().__init__()
        self.N_samples = N_samples #Number of GRF samples
        self.l = np.array(l) #Length scale of GRF
        self.d = len(l) #Dimensionality of GRF
        self.n_mus = int(np.ceil(1/np.prod(self.l))) #Number of GRF sample locations
        self.mus = self.compute_mus()
        self.f_hat = self.compute_coeffs()

    #Sample n_mus random points on the unit square
    def compute_mus(self):
        mus = np.random.uniform(0,1,size=(self.N_samples,self.n_mus,self.d))
        return mus

    #Compute Gaussian covariance matrix cov between points
    def compute_cov(self):
        scaled_mus = self.mus / self.l[None, None, :]
        diff = scaled_mus[:, None, :, :] - scaled_mus[:, :, None, :]
        dist_sq = np.sum(diff**2, axis=-1)
        return np.exp(-0.5 * dist_sq)
    
    def compute_coeffs(self):
        cov = self.compute_cov()
        L = np.linalg.cholesky(cov)
        z = np.random.normal(size=(self.N_samples, self.n_mus))
        f = opt_einsum.contract('Nij,Nj->Ni', L, z)
        L_inv = np.linalg.inv(L)
        cov_inv = opt_einsum.contract('Nij,Njk->Nik', L_inv.transpose(0,2,1), L_inv)
        f_hat = opt_einsum.contract('Nij,Nj->Ni', cov_inv, f)
        return f_hat

    def phi_n(self, i, x):
        # return phi_n_jit(self.f_hat, self.mus, self.l, i, x)
        output = self.f_hat[i,None,:]*np.exp(-1/2*np.sum(((x[:,None,:] - self.mus[i,:,:])/self.l[None,None,:])**2, axis=-1))
        return output

    #Forward evaluation of RBF interpolated GRF
    def forward(self, i):
        def function(x):
            phi_n = self.phi_n(i, x)
            return np.sum(phi_n, axis=1)
        return function
    def forward_x(self, i, x):        
        phi_n = self.phi_n(i, x)
        return np.sum(phi_n, axis=1)        

    # def forward(self, i):
    #     def function(x):
    #         return forward_jax(x, self.f_hat[i], self.mus[i], self.l)
    #     return function

    #Pointwise forward evaluation of RBF interpolated GRF (required for Nutils)
    def forward_nutils(self, i):
        def function(x):
            phi_n = self.f_hat[i]*np.exp(-np.sum((x - self.mus)**2, axis=-1)/(2*self.l**2))
            return np.sum(phi_n)
        return function


    # Gradient of RBF interpolated GRF
    def grad(self, i):
        def function(x):
            phi_n = self.phi_n(i, x)
            prefactor = -1/(self.l[None,None,:]**2)*(x[:,None,:] - self.mus[i,:,:])
            return np.sum(prefactor*phi_n[:,:,None], axis=1)
        return function
    def grad_x(self, i, x):     
        return grad_x_jit(self.f_hat, self.mus, self.l, i, x)   
        # phi_n = self.phi_n(i, x)
        # prefactor = -1/(self.l[None,None,:]**2)*(x[:,None,:] - self.mus[i,:,:])
        # return np.sum(prefactor*phi_n[:,:,None], axis=1)        

    # def grad(self, i):
    #     def function(x):
    #         return grad_numba(x, self.mus[i], self.f_hat[i], self.l)
    #     return function

    # Laplacian of RBF interpolated GRF
    def d2dxi2(self, i):
        def function(x):
            phi_n = self.phi_n(i, x)
            prefactor = (x[:,None,:] - self.mus[i,:,:])**2/self.l[None,None,:]**4 - 1/self.l[None,None,:]**2
            return np.sum(prefactor*phi_n[:,:,None], axis=1)
        return function

    def d2dxi2_x(self, i,x):      
        return d2dxi2_jit(self.f_hat, self.mus, self.l, i, x)  
        # phi_n = self.phi_n(i, x)
        # prefactor = (x[:,None,:] - self.mus[i,:,:])**2/self.l[None,None,:]**4 - 1/self.l[None,None,:]**2
        # return np.sum(prefactor*phi_n[:,:,None], axis=1)
        
    # def d2dxi2(self, i):
    #     def function(x):
    #         return d2dxi2_numba(x, self.mus[i], self.f_hat[i], self.l)
    #     return function  

class ScaledGRF:
    def __init__(self, N_samples, l, c, b):
        super().__init__()
        self.grf = GRF(N_samples, l)
        #Scaling and translation of GRF f'(x) = c f(x) + b
        self.c = c
        self.b = b
    
    def forward(self, i):
        def function(x):
            return self.c[i]*self.grf.forward(i)(x) + self.b[i]
        return function
    def forward_x(self, i, x):        
        return self.c[i]*self.grf.forward_x(i,x) + self.b[i]
        
    
    def forward_nutils(self, i):
        def function(x):
            return self.c[i]*self.grf.forward_nutils(i)(x) + self.b[i]
        return function
    
    def grad(self, i):
        def function(x):
            return self.c[i]*self.grf.grad(i)(x)
        return function
    
    def grad_x(self, i , x ):
        return self.c[i]*self.grf.grad_x(i,x)
    
    def d2dxi2(self, i):
        def function(x):
            return self.c[i]*self.grf.d2dxi2(i)(x)
        return function

    def d2dxi2_x(self, i,x):
        return self.c[i]*self.grf.d2dxi2_x(i,x)