import numpy as np
from scipy.spatial import distance_matrix


class GRF():
    def __init__(self, d, l):
        super().__init__()
        self.d = d
        self.l = l
        self.N_gridpoints = int(1/self.l) + 1
        self.x_grid = self.compute_grid()
        self.cov = self.compute_cov()
        self.f = self.compute_GRFpoints()
        self.f_hat = self.compute_RBFintcoeffs()
        
    def compute_grid(self):
        if self.d==1:
            x_grid = np.linspace(0,1,self.N_gridpoints).reshape((self.N_gridpoints,1))
            print(x_grid)
        if self.d==2:
            X, Y = np.mgrid[0:1:self.N_gridpoints*1j, 0:1:self.N_gridpoints*1j]
            x_grid = np.vstack([X.ravel(), Y.ravel()]).T
        return x_grid
    
    def compute_cov(self):
        cov = np.exp(-distance_matrix(self.x_grid, self.x_grid, p=2)**2/(self.l**2))
        return cov
    
    def compute_GRFpoints(self):
        f = np.random.multivariate_normal(np.zeros(self.N_gridpoints**self.d), cov=self.cov, size=1)[0]
        return f
    
    def compute_RBFintcoeffs(self):
        f_hat = np.dot(np.linalg.inv(self.cov),self.f)
        return f_hat
    
    def RBFinterpolation(self, x):
        if self.d==1:
            terms = self.f_hat*np.exp(-np.sum((x[:,None,None] - self.x_grid[None,:,:])**2, axis=-1)/(self.l**2))
        if self.d==2:
            terms = self.f_hat*np.exp(-np.sum((x[:,None,:] - self.x_grid[None,:,:])**2, axis=-1)/(self.l**2))
        return np.sum(terms, axis=1)

    def RBFint_pointwise(self, x):
        terms = self.f_hat*np.exp(-np.sum((x - self.x_grid)**2, axis=-1)/(self.l**2))
        return np.sum(terms)