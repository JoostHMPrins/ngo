import numpy as np
from scipy.spatial import distance_matrix


class GRF():
    def __init__(self, **kwargs):
        super().__init__()
        self.d = kwargs['d']
        self.l = kwargs['l']
        self.lowerbound = kwargs['lowerbound']
        self.upperbound = kwargs['upperbound']
        self.compute_grid()
        self.compute_cov()
        self.compute_GRFpoints()
        self.compute_RBFintcoeffs()
        if self.lowerbound!=None or self.upperbound!=None:
            self.compute_minmax()
        
    def compute_grid(self):
        # if self.d==2:
        #     X, Y = np.mgrid[0:1:self.N_gridpoints*1j, 0:1:self.N_gridpoints*1j]
        #     self.x_grid = np.vstack([X.ravel(), Y.ravel()]).T
        self.x_grid = np.random.uniform(0,1,size=(int(1/self.l**self.d), self.d))

    def compute_cov(self):
        self.cov = np.exp(-distance_matrix(self.x_grid, self.x_grid, p=2)**2/(2*self.l**2))
    
    def compute_GRFpoints(self):
        self.f = np.random.multivariate_normal(np.zeros(int(1/self.l**self.d)), cov=self.cov)
    
    def compute_RBFintcoeffs(self):
        cov_inv = np.linalg.inv(self.cov)
        self.f_hat = np.einsum('ij,j->i', cov_inv, self.f)
    
    def compute_minmax(self):
        # if self.d==2:
        #     X, Y = np.mgrid[0:1:100*1j, 0:1:100*1j]
        #     x = np.vstack([X.ravel(), Y.ravel()]).T
        x = np.random.uniform(0,1,size=(int((3/self.l)**self.d), self.d))
        terms = self.f_hat[:,None]*np.exp(-np.sum((x[None,:,:] - self.x_grid[:,None,:])**2, axis=-1)/(2*self.l**2))
        f = np.sum(terms, axis=0)
        if self.lowerbound!=None:
            self.f_min = np.amin(f)
        if self.upperbound!=None:
            self.f_max = np.amax(f)
            
    def RBFint(self):
        def function(x):
            terms = self.f_hat[:,None]*np.exp(-np.sum((x[None,:,:] - self.x_grid[:,None,:])**2, axis=-1)/(2*self.l**2))
            output = np.sum(terms, axis=0)
            return output
        return function

    def RBFint_pointwise(self):
        def function(x):
            terms = self.f_hat*np.exp(-np.sum((x - self.x_grid)**2, axis=-1)/(2*self.l**2))
            output = np.sum(terms)
            return output
        return function
    
    def RBFint_scaled(self):
        def function(x):
            if self.lowerbound==None and self.upperbound==None:
                output_scaled = self.RBFint()(x)
            if self.lowerbound!=None and self.upperbound==None:
                output_scaled = self.RBFint()(x) - self.f_min + self.lowerbound
            if self.lowerbound!=None and self.upperbound!=None:
                output_scaled = (self.RBFint()(x) - self.f_min)/(self.f_max - self.f_min) #Scale to [0,1]
                output_scaled = (self.upperbound - self.lowerbound)*output_scaled + self.lowerbound #Scale to [lowerbound,upperbound]
            return output_scaled
        return function
            
    def RBFint_pointwise_scaled(self):
        def function(x):
            if self.lowerbound==None and self.upperbound==None:
                output_scaled = self.RBFint_pointwise()(x)
            if self.lowerbound!=None and self.upperbound==None:
                output_scaled = self.RBFint_pointwise()(x) - self.f_min + self.lowerbound
            if self.lowerbound!=None and self.upperbound!=None:
                output_scaled = (self.RBFint_pointwise()(x) - self.f_min)/(self.f_max - self.f_min) #Scale to [0,1]
                output_scaled = (self.upperbound - self.lowerbound)*output_scaled + self.lowerbound #Scale to [lowerbound,upperbound]
            return output_scaled
        return function
    
    
class GRFset():
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.generate_grfset()
        
    def generate_grfset(self):
        
        grfs_ngo = []
        grfs_nutils = []
        
        for i in range(self.kwargs['N_samples']):
            grf = GRF(**self.kwargs)
            grf_ngo = grf.RBFint_scaled()
            grf_nutils = grf.RBFint_pointwise_scaled()
            grfs_ngo.append(grf_ngo)
            grfs_nutils.append(grf_nutils)
            
        self.grfs_ngo = grfs_ngo
        self.grfs_nutils = grfs_nutils