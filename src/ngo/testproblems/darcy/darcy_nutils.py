# Copyright 2025 Joost Prins

# 3rd Party

from nutils import mesh, function, solver
from nutils.expression_v2 import Namespace
import numpy as np

# Local
from ngo.trainingdata.GRF import ScaledGRF


def Darcy_Nutils(theta, f, etat, etab, gl, gr):
    #Unit square geometry and mesh
    domain, geom = mesh.unitsquare(nelems=20, etype='square')
    #Namespace
    ns = Namespace()
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.basis = domain.basis('spline', degree=3) #Basis
    ns.u = function.dotarg('lhs', ns.basis) #Solution
    #Inputs
    ns.theta = theta(ns.x) #Conductivity
    ns.f = f(ns.x) #Forcing
    ns.etat = etat(ns.x) #Neumann BC top
    ns.etab = etab(ns.x) #Neumann BC bottom
    ns.gl = gl(ns.x) #Dirichlet BC left
    ns.gr = gr(ns.x) #Dirichlet BC right
    #Residual
    res = domain.integral('∇_i(basis_n) theta ∇_i(u) dV' @ ns, degree=6) #Stiffness
    res -= domain.integral('basis_n f dV' @ ns, degree=6) #Forcing
    res -= domain.boundary['top'].integral('basis_n etat dS' @ ns, degree=6) #Neumann BC top
    res -= domain.boundary['bottom'].integral('basis_n etab dS' @ ns, degree=6) #Neumann BC bottom
    #Dirichlet BC
    sqr = domain.boundary['left'].integral('(u - gl)^2 dS' @ ns, degree=6) #Dirichlet BC left
    sqr += domain.boundary['right'].integral('(u - gr)^2 dS' @ ns, degree=6) #Dirichlet BC right
    cons = solver.optimize('lhs', sqr, droptol=1e-15) #Constraints
    #Solve system
    lhs = solver.solve_linear('lhs', res, constrain=cons) #Constrained optimization
    #Sampling of the input functions and solution
    bezier = domain.sample('bezier', 10)
    x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)
    return x, u


class NutilsSetDarcy:
    def __init__(self, N_samples, d, l_min, l_max):
        super().__init__()
        self.N_samples = N_samples #Number of samples
        self.d = d #Dimensionality of problem
        self.l_min = l_min #Minimum GRF length scale
        self.l_max = l_max #Maximum GRF length scale
        self.sensornodes = np.random.uniform(0,1,size=(10000,self.d))
        self.generate_input_functions()
        self.evaluate_input_functions_at_sensors()
        self.generate_nutils_outputs()
    
    def generate_input_functions(self):
        #Empty lists to be filled with functions
        thetas_nutils = []
        fs_nutils = []
        etabs_nutils = []
        etats_nutils = []
        gls_nutils = []
        grs_nutils = []
        thetas = []
        fs = []
        etabs = []
        etats = []
        gls = []
        grs = []
        #Lists of length scales l, scaling factors c and offsets b
        l = np.random.uniform(self.l_min/np.sqrt(2),self.l_max/np.sqrt(2), size=self.N_samples)
        c = np.random.uniform(0,1, size=self.N_samples)
        b = np.random.uniform(-1,1, size=self.N_samples)
        c_theta = np.random.uniform(0,0.2, size=self.N_samples)
        b_theta = np.ones(self.N_samples)
        if self.l_min==self.l_max:
            #Generate batches of GRFs
            theta = ScaledGRF(N_samples=self.N_samples, d=self.d, l=self.l_min, c=c_theta, b=b_theta)
            f = ScaledGRF(N_samples=self.N_samples, d=self.d, l=self.l_min, c=c, b=b)
            etab = ScaledGRF(N_samples=self.N_samples, d=self.d, l=self.l_min, c=c, b=b)
            etat = ScaledGRF(N_samples=self.N_samples, d=self.d, l=self.l_min, c=c, b=b)
            gl = ScaledGRF(N_samples=self.N_samples, d=self.d, l=self.l_min, c=c, b=b)
            gr = ScaledGRF(N_samples=self.N_samples, d=self.d, l=self.l_min, c=c, b=b)
            for i in range(self.N_samples):
                #Collect functions
                thetas_nutils.append(theta.forward_nutils(i))
                fs_nutils.append(f.forward_nutils(i))
                etabs_nutils.append(etab.forward_nutils(i))
                etats_nutils.append(etat.forward_nutils(i))
                gls_nutils.append(gl.forward_nutils(i))
                grs_nutils.append(gr.forward_nutils(i))
                thetas.append(theta.forward(i))
                fs.append(f.forward(i))
                etabs.append(etab.forward(i))
                etats.append(etat.forward(i))
                gls.append(gl.forward(i))
                grs.append(gr.forward(i))
        if self.l_min!=self.l_max:
            for i in range(self.N_samples):
                #Define functions
                theta = ScaledGRF(N_samples=1, d=self.d, l=l[i], c=[c_theta[i]], b=[b_theta[i]])
                f = ScaledGRF(N_samples=1, d=self.d, l=l[i], c=[c[i]], b=[b[i]])
                etab = ScaledGRF(N_samples=1, d=self.d, l=l[i], c=[c[i]], b=[b[i]])
                etat = ScaledGRF(N_samples=1, d=self.d, l=l[i], c=[c[i]], b=[b[i]])
                gl = ScaledGRF(N_samples=1, d=self.d, l=l[i], c=[c[i]], b=[b[i]])
                gr = ScaledGRF(N_samples=1, d=self.d, l=l[i], c=[c[i]], b=[b[i]])
                #Collect functions
                thetas_nutils.append(theta.forward_nutils(0))
                fs_nutils.append(f.forward_nutils(0))
                etabs_nutils.append(etab.forward_nutils(0))
                etats_nutils.append(etat.forward_nutils(0))
                gls_nutils.append(gl.forward_nutils(0))
                grs_nutils.append(gr.forward_nutils(0))
                thetas.append(theta.forward(0))
                fs.append(f.forward(0))
                etabs.append(etab.forward(0))
                etats.append(etat.forward(0))
                gls.append(gl.forward(0))
                grs.append(gr.forward(0))
        #Save functions
        self.theta_nutils = thetas_nutils
        self.f_nutils = fs_nutils
        self.etab_nutils = etabs_nutils
        self.etat_nutils = etats_nutils
        self.gl_nutils = gls_nutils
        self.gr_nutils = grs_nutils
        self.theta = thetas
        self.f = fs
        self.etab = etabs
        self.etat = etats
        self.gl = gls
        self.gr = grs

    def evaluate_input_functions_at_sensors(self):
        self.theta_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.f_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.etab_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.etat_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.gl_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        self.gr_sensor = np.zeros((self.N_samples,self.sensornodes.shape[0]))
        #Evaluate input functions at sensor nodes
        for i in range(self.N_samples):
            self.theta_sensor[i] = self.theta[i](self.sensornodes)
            self.f_sensor[i] = self.f[i](self.sensornodes)
            self.etab_sensor[i] = self.etab[i](self.sensornodes)
            self.etat_sensor[i] = self.etat[i](self.sensornodes)
            self.gl_sensor[i] = self.gl[i](self.sensornodes)
            self.gr_sensor[i] = self.gr[i](self.sensornodes)

    def generate_nutils_outputs(self):
        us = []
        for i in range(self.N_samples):
            x, u = Darcy_Nutils(self.theta_nutils[i], self.f_nutils[i], self.etat_nutils[i], self.etab_nutils[i], self.gl_nutils[i], self.gr_nutils[i])
            us.append(u)
        self.x = np.array(x)
        self.u = np.array(us)
