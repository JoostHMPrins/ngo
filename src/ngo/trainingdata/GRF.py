import numpy as np
from scipy.spatial import distance_matrix
import opt_einsum


class GRF:
    def __init__(self, N_samples, l):
        super().__init__()
        self.N_samples = N_samples #Number of GRF samples
        self.l = np.array(l) #Length scale of GRF
        self.d = len(l) #axisensionality of GRF
        self.n_mus = int(max(10,1/np.prod(self.l))) #Number of GRF sample locations per volume element 1/l^d
        self.mus = self.compute_mus()
        self.f_hat = self.compute_RBFintcoeffs()
        self.mus = np.array(self.mus)

    #Sample n_samples_per_l/l^d random points on the unit square
    def compute_mus(self):
        mus = np.random.uniform(0,1,size=(self.n_mus,self.d))
        return mus

    #Compute Gaussian covariance matrix cov between points
    def compute_cov(self):
        # cov = np.ones((self.n_mus,self.n_mus))
        # for i in range(self.d):
        #     cov *= np.exp(-distance_matrix(self.mus[:,i], self.mus[:,i], p=2)**2/(2*self.l[i]**2))
        l = self.l
        cov = np.exp(-1/2*distance_matrix(self.mus/l[None,:], self.mus/l[None,:], p=2)**2)
        return cov
    
    #Sample from a multivariate Gaussian with covariance cov
    def compute_GRFpoints(self, cov):
        f = np.random.multivariate_normal(np.zeros(self.n_mus), cov=cov, size=self.N_samples)
        return f
    
    #Interpolate the GRF with Gaussian RBFs
    def compute_RBFintcoeffs(self):
        cov = self.compute_cov()
        f = self.compute_GRFpoints(cov)
        cov = np.array(cov)
        cov_inv = np.linalg.inv(cov)
        f = np.array(f)
        f_hat = opt_einsum.contract('ij,nj->ni', cov_inv, f)
        return f_hat
    
    def phi_n(self, i, x):
        output = self.f_hat[i,None,:]*np.exp(-1/2*np.sum(((x[:,None,:] - self.mus[None,:,:])/self.l[None,None,:])**2, axis=-1))
        return output
    
    #Forward evaluation of RBF interpolated GRF
    def forward(self, i):
        def function(x):
            phi_n = self.phi_n(i, x)
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
            phi_n = self.phi_n(i, x)
            prefactor = -1/(self.l[None,None,:]**2)*(x[:,None,:] - self.mus[None,:,:])
            return np.sum(prefactor*phi_n[:,:,None], axis=1)
        return function

    #Laplacian of RBF interpolated GRF
    def d2dxi2(self, i):
        def function(x):
            phi_n = self.phi_n(i, x)
            prefactor = (x[:,None,:] - self.mus[None,:,:])**2/self.l[None,None,:]**4 - 1/self.l[None,None,:]**2
            return np.sum(prefactor*phi_n[:,:,None], axis=1)
        return function
        

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
    
    def forward_nutils(self, i):
        def function(x):
            return self.c[i]*self.grf.forward_nutils(i)(x) + self.b[i]
        return function
    
    def grad(self, i):
        def function(x):
            return self.c[i]*self.grf.grad(i)(x)
        return function
    
    def d2dxi2(self, i):
        def function(x):
            return self.c[i]*self.grf.d2dxi2(i)(x)
        return function