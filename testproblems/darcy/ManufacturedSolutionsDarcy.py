import opt_einsum

import sys
sys.path.insert(0, '/home/prins/st8/prins/phd/gitlab/ngo-pde-gk/trainingdata') 
from GRF import *

class Forcing:
    def __init__(self, theta, u):
        super().__init__()
        self.theta = theta
        self.u = u
    
    def forward(self, i):
        def function(x):
            return -self.theta.forward(i)(x)*self.u.laplacian(i)(x) - np.sum(self.theta.grad(i)(x)*self.u.grad(i)(x), axis=-1)
        return function
    
class NeumannBC:
    def __init__(self, n, theta, u):
        super().__init__()
        self.n = n
        self.theta = theta
        self.u = u

    def forward(self, i):
        def function(x):
            return opt_einsum.contract('i,N,Ni->N', self.n, self.theta.forward(i)(x), self.u.grad(i)(x))
        return function

class DirichletBC:
    def __init__(self, theta, u):
        super().__init__()
        self.theta = theta
        self.u = u

    def forward(self, i):
        def function(x):
            return self.theta.forward(i)(x)*self.u.forward(i)(x)
        return function

class MFSetDarcy:
    def __init__(self, N_samples, d, l_min, l_max):
        super().__init__()
        self.N_samples = N_samples
        self.d = d
        self.l_min = l_min
        self.l_max = l_max
        self.generate_dataset()
    
    def generate_dataset(self):
        thetas = []
        us = []
        fs = []
        etabs = []
        etats = []
        gls = []
        grs = []
        l_theta = np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2), size=self.N_samples)
        c_theta = np.random.uniform(0,0.2,size=self.N_samples)
        b_theta = np.ones(self.N_samples)
        l_u = np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2), size=self.N_samples)
        c_u = np.random.uniform(0,1, size=self.N_samples)
        b_u = np.random.uniform(-1,1, size=self.N_samples)
        if self.l_min==self.l_max:
            #Generate batches of GRFs
            theta = ScaledGRF(N_samples=self.N_samples, d=self.d,l=self.l_min,c=c_theta,b=b_theta)
            u = ScaledGRF(N_samples=self.N_samples, d=self.d,l=self.l_min,c=c_u,b=b_u)
            f = Forcing(theta, u)
            etab = NeumannBC(np.array([0,-1]), theta, u)
            etat = NeumannBC(np.array([0,1]), theta, u)
            gl = DirichletBC(theta, u)
            gr = DirichletBC(theta, u)
            for i in range(self.N_samples):
                #Collect functions
                thetas.append(theta.forward(i))
                us.append(u.forward(i))
                fs.append(f.forward(i))
                etabs.append(etab.forward(i))
                etats.append(etat.forward(i))
                gls.append(gl.forward(i))
                grs.append(gr.forward(i))
        if self.l_min!=self.l_max:
            for i in range(self.N_samples):
                #Define functions
                theta = ScaledGRF(N_samples=1, d=self.d, l=l_theta[i], c=[c_theta[i]], b=[b_theta[i]])
                u = ScaledGRF(N_samples=1, d=self.d, l=l_u[i], c=[c_u[i]], b=[b_u[i]])
                f = Forcing(theta, u)
                etab = NeumannBC(np.array([0,-1]), theta, u)
                etat = NeumannBC(np.array([0,1]), theta, u)
                gl = DirichletBC(theta, u)
                gr = DirichletBC(theta, u)
                #Collect functions
                thetas.append(theta.forward(0))
                us.append(u.forward(0))
                fs.append(f.forward(0))
                etabs.append(etab.forward(0))
                etats.append(etat.forward(0))
                gls.append(gl.forward(0))
                grs.append(gr.forward(0))
        #Save set
        self.theta = thetas
        self.u = us
        self.f = fs
        self.etab = etabs
        self.etat = etats
        self.gl = gls
        self.gr = grs