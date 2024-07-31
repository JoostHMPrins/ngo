import opt_einsum

import sys
sys.path.insert(0, '/home/prins/st8/prins/phd/gitlab/ngo-pde-gk/trainingdata') 
from GRF import *

class Forcing:
    def __init__(self, theta, u):
        super().__init__()
        self.theta = theta
        self.u = u

    def forward(self, x):
        return -self.theta.forward(x)*self.u.laplacian(x) - np.sum(self.theta.grad(x)*self.u.grad(x), axis=-1)
    
class NeumannBC:
    def __init__(self, n, theta, u):
        super().__init__()
        self.n = n
        self.theta = theta
        self.u = u

    def forward(self, x):
        return opt_einsum.contract('i,N,Ni->N', self.n, self.theta.forward(x), self.u.grad(x))
    
class DirichletBC:
    def __init__(self, theta, u):
        super().__init__()
        self.theta = theta
        self.u = u

    def forward(self, x):
        return self.theta.forward(x)*self.u.forward(x)
    

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
        if self.l_min==self.l_max:
            #Generate batches of GRFs
            theta = ScaledGRF(N_samples=self.N_samples, d=self.d,l=np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2)),c=np.random.uniform(0,0.2),b=1)
            u = ScaledGRF(N_samples=self.N_samples, d=self.d,l=np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2)),c=np.random.uniform(),b=np.random.uniform(-1,1))
            for i in range(self.N_samples):
                #Define functions
                theta.i = i
                u.i = i
                f = Forcing(theta, u)
                etab = NeumannBC(np.array([0,-1]),theta,u)
                etat = NeumannBC(np.array([0,1]),theta,u)
                gl = DirichletBC(theta, u)
                gr = DirichletBC(theta, u)
                thetas.append(theta.forward)
                us.append(u.forward)
                #Collect functions
                thetas.append(theta.forward)
                us.append(u.forward)
                fs.append(f.forward)
                etabs.append(etab.forward)
                etats.append(etat.forward)
                gls.append(gl.forward)
                grs.append(gr.forward)
        else:
            for i in range(self.N_samples):
                #Define functions
                # theta = ScaledSquaredGRF(d=self.d,l=np.random.uniform(self.l_min,self.l_max),c=0.2,b=0.5)
                # u = ScaledGRF(d=self.d,l=np.random.uniform(self.l_min/2,self.l_max/2),c=0.01,b=0)
                theta = ScaledGRF(N_samples=1, d=self.d,l=np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2)),c=np.random.uniform(0,0.2),b=1)
                u = ScaledGRF(N_samples=1, d=self.d,l=np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2)),c=np.random.uniform(),b=np.random.uniform(-1,1))
                f = Forcing(theta, u)
                etab = NeumannBC(np.array([0,-1]),theta,u)
                etat = NeumannBC(np.array([0,1]),theta,u)
                gl = DirichletBC(theta, u)
                gr = DirichletBC(theta, u)
                #Collect functions
                thetas.append(theta.forward)
                us.append(u.forward)
                fs.append(f.forward)
                etabs.append(etab.forward)
                etats.append(etat.forward)
                gls.append(gl.forward)
                grs.append(gr.forward)
        #Save set
        self.theta = thetas
        self.u = us
        self.f = fs
        self.etab = etabs
        self.etat = etats
        self.gl = gls
        self.gr = grs