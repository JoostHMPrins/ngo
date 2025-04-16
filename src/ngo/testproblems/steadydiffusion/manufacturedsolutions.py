# Copyright 2025 Joost Prins

# 3rd Party
import numpy as np
import opt_einsum

#Local
from ngo.trainingdata.GRF import ScaledGRF


class Forcing:
    """
    Class to compute the forcing term in the PDE.

    Attributes:
        theta (ScaledGRF): The theta function (diffusion coefficient).
        u (ScaledGRF): The u function (solution).
    """

    def __init__(self, theta, u):
        """
        Initialize the Forcing class.

        Args:
            theta (ScaledGRF): The theta function (diffusion coefficient).
            u (ScaledGRF): The u function (solution).
        """
        super().__init__()
        self.theta = theta
        self.u = u
    
    def forward(self, i):
        """
        Compute the forcing function for the i-th sample.

        Args:
            i (int): Index of the sample.

        Returns:
            function: A function that computes the forcing term at a given point x.
        """
        def function(x):
            """
            Compute the forcing term at a given point x.

            Args:
                x (np.ndarray): Input point, shape (N_points,dimension).

            Returns:
                np.ndarray: Forcing term, shape (x.shape[0]).
            """
            return -self.theta.forward(i)(x)*np.sum(self.u.d2dxi2(i)(x), axis=-1) - np.sum(self.theta.grad(i)(x)*self.u.grad(i)(x), axis=-1)
        return function
    
    
class NeumannBC:
    """
    Class to compute the Neumann boundary condition.

    Attributes:
        n (np.ndarray): Normal vector, shape (d,).
        theta (ScaledGRF): The theta function (diffusion coefficient).
        u (ScaledGRF): The u function (solution).
    """

    def __init__(self, n, theta, u):
        """
        Initialize the NeumannBC class.

        Args:
            n (np.ndarray): Normal vector, shape (d,).
            theta (ScaledGRF): The theta function (diffusion coefficient).
            u (ScaledGRF): The u function (solution).
        """
        super().__init__()
        self.n = n
        self.theta = theta
        self.u = u

    def forward(self, i):
        """
        Compute the Neumann boundary condition for the i-th sample.

        Args:
            i (int): Index of the sample.

        Returns:
            function: A function that computes the Neumann boundary condition at a given point x.
        """
        def function(x):
            """
            Compute the Neumann boundary condition at a given point x.

            Args:
                x (np.ndarray): Input point, shape (N_points,dimension).

            Returns:
                np.ndarray: Neumann boundary condition, shape (x.shape[0]).
            """
            return opt_einsum.contract('i,N,Ni->N', self.n, self.theta.forward(i)(x), self.u.grad(i)(x))
        return function
    

class DirichletBC:
    """
    Class to compute the Dirichlet boundary condition.

    Attributes:
        theta (ScaledGRF): The theta function (diffusion coefficient).
        u (ScaledGRF): The u function (solution).
    """
    def __init__(self, theta, u):
        """
        Initialize the DirichletBC class.

        Args:
            theta (ScaledGRF): The theta function (diffusion coefficient).
            u (ScaledGRF): The u function (solution).
        """
        super().__init__()
        self.theta = theta
        self.u = u

    def forward(self, i):
        """
        Compute the Dirichlet boundary condition for the i-th sample.

        Args:
            i (int): Index of the sample.

        Returns:
            function: A function that computes the Neumann boundary condition at a given point x.
        """
        def function(x):
            """
            Compute the Dirichlet boundary condition at a given point x.

            Args:
                x (np.ndarray): Input point, shape (N_points,dimension).

            Returns:
                np.ndarray: Neumann boundary condition, shape (x.shape[0]).
            """
            return self.theta.forward(i)(x)*self.u.forward(i)(x)
        return function


class ManufacturedSolutionsSet:
    """
    Class to generate a set of manufactured solutions for the PDE.

    Attributes:
        N_samples (int): Number of samples.
        variables (list): List of variables ('t' for temporal variable, 'x' for spatial variable)
        d (int): Dimensionality of the problem.
        l_min (float): Minimum GRF length scale.
        l_max (float): Maximum GRF length scale.
        theta (list): List of theta functions.
        u (list): List of u functions.
        f (list): List of forcing functions.
        eta_y0 (list): List of Neumann boundary conditions at y0.
        eta_yL (list): List of Neumann boundary conditions at yL.
        g_x0 (list): List of Dirichlet boundary conditions at x0.
        g_xL (list): List of Dirichlet boundary conditions at xL.
    """

    def __init__(self, N_samples, variables, l_min, l_max):
        """
        Initialize the ManufacturedSolutionsSet class.

        Args:
            N_samples (int): Number of samples.
            variables (list): List of variables.
            l_min (float): Minimum GRF length scale.
            l_max (float): Maximum GRF length scale.
        """

        super().__init__()
        self.N_samples = N_samples #Number of samples
        self.variables = variables
        self.d = len(l_min) #axisensionality of problem
        self.l_min = l_min #Minimum GRF length scale
        self.l_max = l_max #Maximum GRF length scale
        self.generate_manufactured_solutions()

    def generate_manufactured_solutions(self):
        """
        Generate the manufactured solutions.

        Returns:
            None
        """
        #Empty lists to be filled with functions
        thetas = []
        us = []
        fs = []
        eta_y0s = []
        eta_yLs = []
        g_x0s = []
        g_xLs = []
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
        n_b = np.array([0,-1])
        n_t = np.array([0,1])
        if self.l_min==self.l_max:
            #Generate batches of GRFs with the same length scale
            theta = ScaledGRF(N_samples=self.N_samples, l=self.l_min, c=c_theta, b=b_theta)
            u = ScaledGRF(N_samples=self.N_samples, l=self.l_min, c=c_u, b=b_u)
            f = Forcing(theta, u)
            eta_y0 = NeumannBC(n_b, theta, u)
            eta_yL = NeumannBC(n_t, theta, u)
            g_x0 = DirichletBC(theta, u)
            g_xL = DirichletBC(theta, u)
            for i in range(self.N_samples):
                #Collect functions
                thetas.append(theta.forward(i))
                us.append(u.forward(i))
                fs.append(f.forward(i))
                eta_y0s.append(eta_y0.forward(i))
                eta_yLs.append(eta_yL.forward(i))
                g_x0s.append(g_x0.forward(i))
                g_xLs.append(g_xL.forward(i))
        if self.l_min!=self.l_max:
            for i in range(self.N_samples):
                #Define functions
                theta = ScaledGRF(N_samples=1, l=l_theta[i], c=[c_theta[i]], b=[b_theta[i]])
                u = ScaledGRF(N_samples=1, l=l_u[i], c=[c_u[i]], b=[b_u[i]])
                f = Forcing(theta, u)
                eta_y0 = NeumannBC(n_b, theta, u)
                eta_yL = NeumannBC(n_t, theta, u)
                g_x0 = DirichletBC(theta, u)
                g_xL = DirichletBC(theta, u)
                #Collect functions
                thetas.append(theta.forward(0))
                us.append(u.forward(0))
                fs.append(f.forward(0))
                eta_y0s.append(eta_y0.forward(0))
                eta_yLs.append(eta_yL.forward(0))
                g_x0s.append(g_x0.forward(0))
                g_xLs.append(g_xL.forward(0))
        #Save set
        self.theta = thetas
        self.u = us
        self.f = fs
        self.eta_y0 = eta_y0s
        self.eta_yL = eta_yLs
        self.g_x0 = g_x0s
        self.g_xL = g_xLs
