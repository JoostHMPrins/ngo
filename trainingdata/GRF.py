import numpy as np
from scipy.spatial import distance_matrix
import opt_einsum

class GRF:
    def __init__(self, N_samples, d, l):
        super().__init__()
        self.N_samples = N_samples #Number of GRF samples
        self.d = d #Dimensionality of GRF
        self.l = l #Length scale of GRF
        self.n_samples_per_l = 1 #Number of GRF sample locations per volume element 1/l^d
        self.mus = self.compute_mus()
        self.f_hat = self.compute_RBFintcoeffs()
    
    #Sample n_samples_per_l/l^d random points on the unit square
    def compute_mus(self):
        mus = np.random.uniform(0,1,size=(int(np.ceil(self.n_samples_per_l/self.l**self.d)), self.d))
        return mus

    #Compute Gaussian covariance matrix cov between points
    def compute_cov(self):
        cov = np.exp(-distance_matrix(self.mus, self.mus, p=2)**2/(2*self.l**2))
        return cov
    
    #Sample from a multivariate Gaussian with covariance cov
    def compute_GRFpoints(self, cov):
        f = np.random.multivariate_normal(np.zeros(int(np.ceil(self.n_samples_per_l/self.l**self.d))), cov=cov, size=self.N_samples)
        return f
    
    #Interpolate the GRF with Gaussian RBFs
    def compute_RBFintcoeffs(self):
        cov = self.compute_cov()
        cov_inv = np.linalg.inv(cov)
        f = self.compute_GRFpoints(cov)
        f_hat = opt_einsum.contract('ij,nj->ni', cov_inv, f)
        return f_hat
    
    #Forward evaluation of RBF interpolated GRF
    def forward(self, i):
        def function(x):
            phi_n = self.f_hat[i,None,:]*np.exp(-np.sum((x[:,None,:] - self.mus[None,:,:])**2, axis=-1)/(2*self.l**2))
            return np.sum(phi_n, axis=1)
        return function

    #Pointwise forward evaluation of RBF interpolated GRF (required for Nutils)
    def forward_nutils(self, i):
        def function(x):
            phi_n = self.f_hat[i]*np.exp(-np.sum((x - self.mus)**2, axis=-1)/(2*self.l**2))
            return np.sum(phi_n)
        return function
    
    #Gradient of RBF interpolated GRF
    def grad(self, i):
        def function(x):
            phi_n = self.f_hat[i,None,:]*np.exp(-np.sum((x[:,None,:] - self.mus[None,:,:])**2, axis=-1)/(2*self.l**2))
            prefactor = -1/(self.l**2)*(x[:,None,:] - self.mus[None,:,:])
            return np.sum(prefactor*phi_n[:,:,None], axis=1)
        return function

    #Laplacian of RBF interpolated GRF
    def laplacian(self, i):
        def function(x):
            phi_n = self.f_hat[i,None,:]*np.exp(-np.sum((x[:,None,:] - self.mus[None,:,:])**2, axis=-1)/(2*self.l**2))
            prefactor = (1/self.l**2)*(np.sum((x[:,None,:] - self.mus[None,:,:])**2, axis=-1)/self.l**2 - self.d)
            return np.sum(prefactor*phi_n, axis=1)
        return function
        

class ScaledGRF:
    def __init__(self, N_samples, d, l, c, b):
        super().__init__()
        self.grf = GRF(N_samples, d, l)
        #Scaling and translation of GRF f'(x) = c f(x) + b
        self.c = c
        self.b = b
    
    def forward(self, i):
        def function(x):
            return self.c[i]*self.grf.forward(i)(x) + self.b[i]
        return function
    
    def forward_nutils(self, i):
        def function(x):
            return self.c[i]*self.grf.forward_nutils(i)(x) + self.b[i]
        return function
    
    def grad(self, i):
        def function(x):
            return self.c[i]*self.grf.grad(i)(x)
        return function
    
    def laplacian(self, i):
        def function(x):
            return self.c[i]*self.grf.laplacian(i)(x)
        return function