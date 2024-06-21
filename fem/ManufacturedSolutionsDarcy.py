from GRF import *
import opt_einsum

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
    def __init__(self, N_samples, d, l_theta_min, l_theta_max, l_u_min, l_u_max):
        super().__init__()
        self.N_samples = N_samples
        self.d = d
        self.l_theta_min = l_theta_min
        self.l_theta_max = l_theta_max
        self.l_u_min = l_u_min
        self.l_u_max = l_u_max
        self.generate_dataset()
    
    def generate_dataset(self):
        thetas = []
        us = []
        fs = []
        etabs = []
        etats = []
        gls = []
        grs = []
        for i in range(self.N_samples):
            #Define functions
            theta = ScaledSquaredGRF(d=self.d,l=np.random.uniform(self.l_theta_min,self.l_theta_max),c=0.2,b=0.5)
            u = ScaledGRF(d=self.d,l=np.random.uniform(self.l_u_min,self.l_u_max),c=0.01,b=0)
            f = Forcing(theta, u)
            etab = NeumannBC(np.array([0,-1]),theta, u)
            etat = NeumannBC(np.array([0,1]),theta, u)
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