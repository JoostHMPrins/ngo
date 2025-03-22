# Copyright 2025 Joost Prins

# 3rd Party
import numpy as np
import numpy
import opt_einsum

# Local
from ngo.trainingdata.GRF import ScaledGRF


class Forcing:
    def __init__(self, theta, u):
        super().__init__()
        self.theta = theta
        self.u = u
    
    def forward(self, i):
        def function(x):
            return -self.theta.forward(i)(x)*numpy.sum(self.u.d2dxi2(i)(x), axis=-1) - numpy.sum(self.theta.grad(i)(x)*self.u.grad(i)(x), axis=-1)
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


class ManufacturedSolutionsSetDarcy:
    def __init__(self, N_samples, variables, l_min, l_max):
        super().__init__()
        self.N_samples = N_samples #Number of samples
        self.variables = variables
        self.d = len(l_min) #axisensionality of problem
        self.l_min = l_min #Minimum GRF length scale
        self.l_max = l_max #Maximum GRF length scale
        self.generate_manufactured_solutions()
        # self.sensornodes = np.random.uniform(0,1,size=(10000,self.d))
        # self.evaluate_functions_at_sensors()

    def generate_manufactured_solutions(self):
        #Empty lists to be filled with functions
        thetas = []
        us = []
        fs = []
        etabs = []
        etats = []
        gls = []
        grs = []
        #Lists of length scales l, scaling factors c and offsets b
        l_theta = np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2), size=(self.N_samples,self.d))
        c_theta = np.random.uniform(0,0.2,size=self.N_samples)
        b_theta = np.ones(self.N_samples)
        l_u = np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2), size=(self.N_samples,self.d))
        c_u = np.random.uniform(0,1, size=self.N_samples)
        b_u = np.random.uniform(-1,1, size=self.N_samples)
        for i in range(1,len(self.variables)):
            if self.variables[i]==self.variables[i-1]:
                l_theta[:,i] = l_theta[:,i-1]
                l_u[:,i] = l_u[:,i-1]
        #Neumann boundary normals
        n_b = numpy.array([0,-1])
        n_t = numpy.array([0,1])
        if self.l_min==self.l_max:
            #Generate batches of GRFs with the same length scale
            theta = ScaledGRF(N_samples=self.N_samples, l=self.l_min, c=c_theta, b=b_theta)
            u = ScaledGRF(N_samples=self.N_samples, l=self.l_min, c=c_u, b=b_u)
            f = Forcing(theta, u)
            etab = NeumannBC(n_b, theta, u)
            etat = NeumannBC(n_t, theta, u)
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
                print(i)
                #Define functions
                theta = ScaledGRF(N_samples=1, l=l_theta[i], c=[c_theta[i]], b=[b_theta[i]])
                u = ScaledGRF(N_samples=1, l=l_u[i], c=[c_u[i]], b=[b_u[i]])
                f = Forcing(theta, u)
                etab = NeumannBC(n_b, theta, u)
                etat = NeumannBC(n_t, theta, u)
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

    def evaluate_functions_at_sensors(self):
        self.theta_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.f_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.etab_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.etat_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.gl_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.gr_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.u_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        #Evaluate input functions at sensor nodes
        for i in range(self.N_samples):
            self.theta_sensor[i] = self.theta[i](self.sensornodes)
            self.f_sensor[i] = self.f[i](self.sensornodes)
            self.etab_sensor[i] = self.etab[i](self.sensornodes)
            self.etat_sensor[i] = self.etat[i](self.sensornodes)
            self.gl_sensor[i] = self.gl[i](self.sensornodes)
            self.gr_sensor[i] = self.gr[i](self.sensornodes)
            self.u_sensor[i] = self.u[i](self.sensornodes)


class ManufacturedSolutionsSetDarcy_ctheta:
    def __init__(self, N_samples, variables, l_min, l_max, c_theta_min, c_theta_max):
        super().__init__()
        self.N_samples = N_samples #Number of samples
        self.variables = variables
        self.d = len(l_min) #axisensionality of problem
        self.l_min = l_min #Minimum GRF length scale
        self.l_max = l_max #Maximum GRF length scale
        self.c_theta_min = c_theta_min
        self.c_theta_max = c_theta_max
        self.generate_manufactured_solutions()
        # self.sensornodes = np.random.uniform(0,1,size=(10000,self.d))
        # self.evaluate_functions_at_sensors()

    def generate_manufactured_solutions(self):
        #Empty lists to be filled with functions
        thetas = []
        us = []
        fs = []
        etabs = []
        etats = []
        gls = []
        grs = []
        #Lists of length scales l, scaling factors c and offsets b
        l_theta = np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2), size=(self.N_samples,self.d))
        c_theta = np.random.uniform(self.c_theta_min,self.c_theta_max,size=self.N_samples)
        b_theta = np.ones(self.N_samples)
        l_u = np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2), size=(self.N_samples,self.d))
        c_u = np.random.uniform(0,1, size=self.N_samples)
        b_u = np.random.uniform(-1,1, size=self.N_samples)
        for i in range(1,len(self.variables)):
            if self.variables[i]==self.variables[i-1]:
                l_theta[:,i] = l_theta[:,i-1]
                l_u[:,i] = l_u[:,i-1]
        #Neumann boundary normals
        n_b = numpy.array([0,-1])
        n_t = numpy.array([0,1])
        if self.l_min==self.l_max:
            #Generate batches of GRFs with the same length scale
            theta = ScaledGRF(N_samples=self.N_samples, l=self.l_min, c=c_theta, b=b_theta)
            u = ScaledGRF(N_samples=self.N_samples, l=self.l_min, c=c_u, b=b_u)
            f = Forcing(theta, u)
            etab = NeumannBC(n_b, theta, u)
            etat = NeumannBC(n_t, theta, u)
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
                theta = ScaledGRF(N_samples=1, l=l_theta[i], c=[c_theta[i]], b=[b_theta[i]])
                u = ScaledGRF(N_samples=1, l=l_u[i], c=[c_u[i]], b=[b_u[i]])
                f = Forcing(theta, u)
                etab = NeumannBC(n_b, theta, u)
                etat = NeumannBC(n_t, theta, u)
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

    def evaluate_functions_at_sensors(self):
        self.theta_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.f_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.etab_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.etat_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.gl_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.gr_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.u_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        #Evaluate input functions at sensor nodes
        for i in range(self.N_samples):
            self.theta_sensor[i] = self.theta[i](self.sensornodes)
            self.f_sensor[i] = self.f[i](self.sensornodes)
            self.etab_sensor[i] = self.etab[i](self.sensornodes)
            self.etat_sensor[i] = self.etat[i](self.sensornodes)
            self.gl_sensor[i] = self.gl[i](self.sensornodes)
            self.gr_sensor[i] = self.gr[i](self.sensornodes)
            self.u_sensor[i] = self.u[i](self.sensornodes)


class ManufacturedSolutionsSetDarcy_bctheta:
    def __init__(self, N_samples, variables, l_min, l_max, b_theta_min, b_theta_max, c_theta_min, c_theta_max):
        super().__init__()
        self.N_samples = N_samples #Number of samples
        self.variables = variables
        self.d = len(l_min) #axisensionality of problem
        self.l_min = l_min #Minimum GRF length scale
        self.l_max = l_max #Maximum GRF length scale
        self.b_theta_min = b_theta_min
        self.b_theta_max = b_theta_max
        self.c_theta_min = c_theta_min
        self.c_theta_max = c_theta_max
        self.generate_manufactured_solutions()
        # self.sensornodes = np.random.uniform(0,1,size=(10000,self.d))
        # self.evaluate_functions_at_sensors()

    def generate_manufactured_solutions(self):
        #Empty lists to be filled with functions
        thetas = []
        us = []
        fs = []
        etabs = []
        etats = []
        gls = []
        grs = []
        #Lists of length scales l, scaling factors c and offsets b
        l_theta = np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2), size=(self.N_samples,self.d))
        c_theta = np.random.uniform(self.c_theta_min,self.c_theta_max,size=self.N_samples)
        b_theta = np.random.uniform(self.b_theta_min,self.b_theta_max,size=self.N_samples)
        l_u = np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2), size=(self.N_samples,self.d))
        c_u = np.random.uniform(0,1, size=self.N_samples)
        b_u = np.random.uniform(-1,1, size=self.N_samples)
        for i in range(1,len(self.variables)):
            if self.variables[i]==self.variables[i-1]:
                l_theta[:,i] = l_theta[:,i-1]
                l_u[:,i] = l_u[:,i-1]
        #Neumann boundary normals
        n_b = numpy.array([0,-1])
        n_t = numpy.array([0,1])
        if self.l_min==self.l_max:
            #Generate batches of GRFs with the same length scale
            theta = ScaledGRF(N_samples=self.N_samples, l=self.l_min, c=c_theta, b=b_theta)
            u = ScaledGRF(N_samples=self.N_samples, l=self.l_min, c=c_u, b=b_u)
            f = Forcing(theta, u)
            etab = NeumannBC(n_b, theta, u)
            etat = NeumannBC(n_t, theta, u)
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
                # print(i)
                #Define functions
                theta = ScaledGRF(N_samples=1, l=l_theta[i], c=[c_theta[i]], b=[b_theta[i]])
                u = ScaledGRF(N_samples=1, l=l_u[i], c=[c_u[i]], b=[b_u[i]])
                f = Forcing(theta, u)
                etab = NeumannBC(n_b, theta, u)
                etat = NeumannBC(n_t, theta, u)
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

    def evaluate_functions_at_sensors(self):
        self.theta_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.f_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.etab_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.etat_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.gl_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.gr_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.u_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        #Evaluate input functions at sensor nodes
        for i in range(self.N_samples):
            self.theta_sensor[i] = self.theta[i](self.sensornodes)
            self.f_sensor[i] = self.f[i](self.sensornodes)
            self.etab_sensor[i] = self.etab[i](self.sensornodes)
            self.etat_sensor[i] = self.etat[i](self.sensornodes)
            self.gl_sensor[i] = self.gl[i](self.sensornodes)
            self.gr_sensor[i] = self.gr[i](self.sensornodes)
            self.u_sensor[i] = self.u[i](self.sensornodes)


class ManufacturedSolutionsSetDarcy_btheta:
    def __init__(self, N_samples, variables, l_min, l_max, b_theta_min, b_theta_max):
        super().__init__()
        self.N_samples = N_samples #Number of samples
        self.variables = variables
        self.d = len(l_min) #axisensionality of problem
        self.l_min = l_min #Minimum GRF length scale
        self.l_max = l_max #Maximum GRF length scale
        self.b_theta_min = b_theta_min
        self.b_theta_max = b_theta_max
        self.generate_manufactured_solutions()
        # self.sensornodes = np.random.uniform(0,1,size=(10000,self.d))
        # self.evaluate_functions_at_sensors()

    def generate_manufactured_solutions(self):
        #Empty lists to be filled with functions
        thetas = []
        us = []
        fs = []
        etabs = []
        etats = []
        gls = []
        grs = []
        #Lists of length scales l, scaling factors c and offsets b
        l_theta = np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2), size=(self.N_samples,self.d))
        c_theta = np.random.uniform(0,0.2,size=self.N_samples)
        b_theta = np.random.uniform(self.b_theta_min,self.b_theta_max,size=self.N_samples)
        l_u = np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2), size=(self.N_samples,self.d))
        c_u = np.random.uniform(0,1, size=self.N_samples)
        b_u = np.random.uniform(-1,1, size=self.N_samples)
        for i in range(1,len(self.variables)):
            if self.variables[i]==self.variables[i-1]:
                l_theta[:,i] = l_theta[:,i-1]
                l_u[:,i] = l_u[:,i-1]
        #Neumann boundary normals
        n_b = np.array([0,-1])
        n_t = numpy.array([0,1])
        if self.l_min==self.l_max:
            #Generate batches of GRFs with the same length scale
            theta = ScaledGRF(N_samples=self.N_samples, l=self.l_min, c=c_theta, b=b_theta)
            u = ScaledGRF(N_samples=self.N_samples, l=self.l_min, c=c_u, b=b_u)
            f = Forcing(theta, u)
            etab = NeumannBC(n_b, theta, u)
            etat = NeumannBC(n_t, theta, u)
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
                # print(i)
                #Define functions
                theta = ScaledGRF(N_samples=1, l=l_theta[i], c=[c_theta[i]], b=[b_theta[i]])
                u = ScaledGRF(N_samples=1, l=l_u[i], c=[c_u[i]], b=[b_u[i]])
                f = Forcing(theta, u)
                etab = NeumannBC(n_b, theta, u)
                etat = NeumannBC(n_t, theta, u)
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

    def evaluate_functions_at_sensors(self):
        self.theta_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.f_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.etab_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.etat_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.gl_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.gr_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.u_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        #Evaluate input functions at sensor nodes
        for i in range(self.N_samples):
            self.theta_sensor[i] = self.theta[i](self.sensornodes)
            self.f_sensor[i] = self.f[i](self.sensornodes)
            self.etab_sensor[i] = self.etab[i](self.sensornodes)
            self.etat_sensor[i] = self.etat[i](self.sensornodes)
            self.gl_sensor[i] = self.gl[i](self.sensornodes)
            self.gr_sensor[i] = self.gr[i](self.sensornodes)
            self.u_sensor[i] = self.u[i](self.sensornodes)