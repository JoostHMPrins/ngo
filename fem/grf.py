import numpy as np
from scipy.spatial import distance_matrix


class GRF():
    def __init__(self, d, l, N_samples, lowerbound, upperbound):
        super().__init__()
        self.d = d
        self.l = l
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.N_samples = N_samples
        self.N_gridpoints = int(1/self.l) + 1
        self.compute_grid()
        self.compute_cov()
        self.compute_GRFpoints()
        self.compute_RBFintcoeffs()
        self.compute_minmax()
        
    def compute_grid(self):
        if self.d==1:
            self.x_grid = np.linspace(0,1,self.N_gridpoints).reshape((self.N_gridpoints,1))
        if self.d==2:
            X, Y = np.mgrid[0:1:self.N_gridpoints*1j, 0:1:self.N_gridpoints*1j]
            self.x_grid = np.vstack([X.ravel(), Y.ravel()]).T
    
    def compute_cov(self):
        self.cov = np.exp(-distance_matrix(self.x_grid, self.x_grid, p=2)**2/(2*self.l**2))
    
    def compute_GRFpoints(self):
        self.f = np.random.multivariate_normal(np.zeros(self.N_gridpoints**self.d), cov=self.cov, size=self.N_samples)
    
    def compute_RBFintcoeffs(self):
        cov_inv = np.linalg.inv(self.cov)
        self.f_hat = np.einsum('nij,nj->ni', np.tile(cov_inv, (self.f.shape[0],1,1)), self.f)
    
    def RBFint(self, x):
        if self.d==1:
            x = x.reshape((len(x),1))
        terms = self.f_hat[:,:,None]*np.exp(-np.sum((x[None,None,:,:] - self.x_grid[None,:,None,:])**2, axis=-1)/(2*self.l**2))
        output = np.sum(terms, axis=1)
        return output

    def RBFint_pointwise(self, sample):
        terms = self.f_hat*np.exp(-np.sum((x - self.x_grid)**2, axis=-1)/(2*self.l**2))
        output = np.sum(terms, axis=1)
        return output
    
    def compute_minmax(self):
        if self.d==1:
            x = np.linspace(0,1,100)
            self.f_min = np.amin(self.RBFint(x))
            self.f_max = np.amax(self.RBFint(x))
        if self.d==2:
            X, Y = np.mgrid[0:1:100*1j, 0:1:100*1j]
            x = np.vstack([X.ravel(), Y.ravel()]).T
            self.f_min = np.amin(self.RBFint(x))
            self.f_max = np.amax(self.RBFint(x))
            
    def RBFint_scaled(self, x):
        output_scaled = (self.RBFint(x) - self.f_min)/(self.f_max - self.f_min) #Scale to [0,1]
        output_scaled = (self.upperbound - self.lowerbound)*output_scaled + self.lowerbound #Scale to [lowerbound,upperbound]
        return output_scaled
            
    def RBFint_pointwise_scaled(self, x):
        output_scaled = (self.RBFint_pointwise(x) - self.f_min)/(self.f_max - self.f_min) #Scale to [0,1]
        output_scaled = (self.upperbound - self.lowerbound)*output_scaled + self.lowerbound #Scale to [lowerbound,upperbound]
        return output_scaled
    
    def sample(self, f, i):
        def function(x):
            return f(x)[i]
        return function  