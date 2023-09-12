import numpy as np
from scipy.spatial import distance_matrix


class GRF():
    def __init__(self, **kwargs):
        super().__init__()
        self.d = kwargs['d']
        #Fixed length scale
        if kwargs['l_min']==kwargs['l_max']:
            self.l = kwargs['l_min']
            self.N_samples = kwargs['N_samples']
        #Random length scale
        if kwargs['l_min']!=kwargs['l_max']:
            self.l = np.random.uniform(kwargs['l_min'], kwargs['l_max'])
            self.N_samples = 2
        self.lowerbound = kwargs['lowerbound']
        self.upperbound = kwargs['upperbound']
        self.N_samples = kwargs['N_samples']
        self.N_gridpoints = int(1/self.l) + 1
        print(0)
        self.compute_grid()
        print(1)
        self.compute_cov()
        print(2)
        self.compute_GRFpoints()
        print(3)
        self.compute_RBFintcoeffs()
        print(4)
        if self.lowerbound!=None or self.upperbound!=None:
            self.compute_minmax()
        print(5)
        
    def compute_grid(self):
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
    
    def compute_minmax(self):
        if self.d==2:
            X, Y = np.mgrid[0:1:100*1j, 0:1:100*1j]
            x = np.vstack([X.ravel(), Y.ravel()]).T
        terms = self.f_hat[:,:,None]*np.exp(-np.sum((x[None,None,:,:] - self.x_grid[None,:,None,:])**2, axis=-1)/(2*self.l**2))
        f = np.sum(terms, axis=1)
        self.f_min = np.amin(f)
        self.f_max = np.amax(f)
            
    def RBFint(self, sample):
        def function(x):
            terms = self.f_hat[sample,:,None]*np.exp(-np.sum((x[None,None,:,:] - self.x_grid[None,:,None,:])**2, axis=-1)/(2*self.l**2))
            output = np.sum(terms, axis=1)
            return output
        return function

    def RBFint_pointwise(self, sample):
        def function(x):
            terms = self.f_hat[sample]*np.exp(-np.sum((x - self.x_grid)**2, axis=-1)/(2*self.l**2))
            output = np.sum(terms)
            return output
        return function
    
    def RBFint_scaled(self, sample):
        def function(x):
            if self.lowerbound==None and self.upperbound==None:
                output_scaled = self.RBFint(sample)(x)
            if self.lowerbound!=None and self.upperbound==None:
                output_scaled = self.RBFint(sample)(x) - self.f_min + self.lowerbound
            if self.lowerbound!=None and self.upperbound!=None:
                output_scaled = (self.RBFint(sample)(x) - self.f_min)/(self.f_max - self.f_min) #Scale to [0,1]
                output_scaled = (self.upperbound - self.lowerbound)*output_scaled + self.lowerbound #Scale to [lowerbound,upperbound]
            return output_scaled
        return function
            
    def RBFint_pointwise_scaled(self, sample):
        def function(x):
            if self.lowerbound==None and self.upperbound==None:
                output_scaled = self.RBFint_pointwise(sample)(x)
            if self.lowerbound!=None and self.upperbound==None:
                output_scaled = self.RBFint_pointwise(sample)(x) - self.f_min + self.lowerbound
            if self.lowerbound!=None and self.upperbound!=None:
                output_scaled = (self.RBFint_pointwise(sample)(x) - self.f_min)/(self.f_max - self.f_min) #Scale to [0,1]
                output_scaled = (self.upperbound - self.lowerbound)*output_scaled + self.lowerbound #Scale to [lowerbound,upperbound]
            return output_scaled
        return function