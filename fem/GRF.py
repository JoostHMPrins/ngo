import numpy as np
from scipy.spatial import distance_matrix
from numba import jit
import opt_einsum
from numba import jit
import scipy


class GRF:
    def __init__(self, d, l):
        super().__init__()
        self.d = d
        self.l = l
        self.mus = self.compute_mus()
        self.f_hat = self.compute_RBFintcoeffs()
        
    def compute_mus(self):
        mus = np.random.uniform(0,1,size=(int(np.ceil(1/self.l**self.d)), self.d))
        return mus

    def compute_cov(self):
        cov = np.exp(-distance_matrix(self.mus, self.mus, p=2)**2/(2*self.l**2))
        return cov
    
    def compute_GRFpoints(self, cov):
        f = np.random.multivariate_normal(np.zeros(int(np.ceil(1/self.l**self.d))), cov=cov)
        return f
    
    def compute_RBFintcoeffs(self):
        cov = self.compute_cov()
        f = self.compute_GRFpoints(cov)
        f_hat = scipy.linalg.lstsq(cov, f, lapack_driver='gelsy', check_finite=False)[0]
        return f_hat
    
    def phi_n(self, x):
        phi_n = self.f_hat[None,:]*np.exp(-np.sum((x[:,None,:] - self.mus[None,:,:])**2, axis=-1)/(2*self.l**2))
        return phi_n
    
    def forward(self, x):
        phi_n = self.phi_n(x)
        return np.sum(phi_n, axis=1)
    
    def grad(self,x):
        phi_n = self.phi_n(x)
        prefactor = -1/(self.l**2)*(x[:,None,:] - self.mus[None,:,:])
        return np.sum(prefactor*phi_n[:,:,None], axis=1)
    
    def laplacian(self, x):
        phi_n = self.phi_n(x)
        prefactor = (1/self.l**2)*(np.sum((x[:,None,:] - self.mus[None,:,:])**2, axis=-1)/self.l**2 - self.d)
        return np.sum(prefactor*phi_n, axis=1)

    
class SquaredGRF:
    def __init__(self, d, l):
        super().__init__()
        self.grf = GRF(d, l)   
        
    def forward(self, x):
        return 1/2*(self.grf.forward(x))**2
    
    def grad(self, x):
        return self.grf.forward(x)[:,None]*self.grf.grad(x)
    
    def laplacian(self, x):
        return self.grf.forward(x)*self.grf.laplacian(x) + np.sum(self.grf.grad(x)**2, axis=-1)
        

class ScaledGRF:
    def __init__(self, d, l, c, b):
        super().__init__()
        self.grf = GRF(d, l)
        self.c = c
        self.b = b
        
    def forward(self, x):
        return self.c*self.grf.forward(x) + self.b
    
    def grad(self, x):
        return self.c*self.grf.grad(x)
    
    def laplacian(self, x):
        return self.c*self.grf.laplacian(x)
    
    
class ScaledSquaredGRF:
    def __init__(self, d, l, c, b):
        super().__init__()
        self.squaredgrf = SquaredGRF(d, l)
        self.c = c
        self.b = b
        
    def forward(self, x):
        return self.c*self.squaredgrf.forward(x) + self.b
    
    def grad(self, x):
        return self.c*self.squaredgrf.grad(x)
    
    def laplacian(self, x):
        return self.c*self.squaredgrf.laplacian(x)