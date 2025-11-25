# Copyright 2025 Joost Prins

# 3rd Party
import torch
import opt_einsum

#Local
from ngo.trainingdata.GRF import ManufacturedSolution
from ngo.numericalanalysis.basisfunctions import GaussianRBFBasis, TensorProductBasis


class Forcing:
    """
    Forcing term for the steady diffusion equation, given a manufactured diffusion coefficient theta and solution u.
    """
    def __init__(self, theta, u):
        """
        Args: 
            theta : ManufacturedSolution
                Manufactured solution representing the diffusion coefficient.
            u : ManufacturedSolution
                Manufactured solution representing the solution to the diffusion equation.
        """
        super().__init__()
        self.theta = theta
        self.u = u
    
    def forward(self, X):
        """
        Forward pass to compute the forcing term at locations X.

        Args:
            X : tuple of torch.Tensor
                Input locations where the forcing term is evaluated, all of shape (Q_i,).

        Returns:
            torch.Tensor
                The computed forcing term at the input locations, shape (num_points_1, num_points_2, ..., num_points_d).
        """
        return self.u.grad(X)[:,:,:,:,0] - self.theta.forward(X)*torch.sum(self.u.laplacian(X)[:,:,:,:,1:], axis=-1) - torch.sum(self.theta.grad(X)[:,:,:,:,1:]*self.u.grad(X)[:,:,:,:,1:], axis=-1)
    

class NeumannBC:
    """
    Neumann boundary condition for the steady diffusion equation, given a manufactured diffusion coefficient theta and solution u.
    """
    def __init__(self, n, theta, u):
        """
        Args:
            n : torch.Tensor
                Normal vectors at the boundary, shape (d, num_boundary_points).
            theta : ManufacturedSolution
                Manufactured solution representing the diffusion coefficient.
            u : ManufacturedSolution
                Manufactured solution representing the solution to the diffusion equation.
        """
        super().__init__()
        self.n = n
        self.theta = theta
        self.u = u

    def forward(self, X):
        """
        Forward pass to compute the Neumann boundary condition at locations X.
        Args:
            X : tuple of torch.Tensor
                Input locations where the Neumann boundary condition is evaluated, all of shape (num_boundary_points_i,).

        Returns:
            torch.Tensor
                The computed Neumann boundary condition at the input locations, shape (num_boundary_points_1, num_boundary_points_2, ..., num_boundary_points_d).
        """
        return opt_einsum.contract('i,N...,N...i->N...', self.n, self.theta.forward(X), self.u.grad(X)[:,:,:,:,1:])
    

class DirichletBC:
    """
    Dirichlet boundary condition for the steady diffusion equation, given a manufactured diffusion coefficient theta and solution u.
    """
    def __init__(self, theta, u):
        """
        Args:
            theta : ManufacturedSolution
                Manufactured solution representing the diffusion coefficient.
            u : ManufacturedSolution
                Manufactured solution representing the solution to the diffusion equation.
        """
        super().__init__()
        self.theta = theta
        self.u = u

    def forward(self, X):
        """
        Forward pass to compute the Dirichlet boundary condition at locations X.

        Args:
            X : tuple of torch.Tensor
                Input locations where the Dirichlet boundary condition is evaluated, all of shape (num_boundary_points_i,).
                
        Returns:
            torch.Tensor
                The computed Dirichlet boundary condition at the input locations, shape (num_boundary_points_1, num_boundary_points_2, ..., num_boundary_points_d).
        """
        return self.theta.forward(X)*self.u.forward(X)
    

class IC:
    """
    Initial condition for the steady diffusion equation, given a manufactured solution u.
    """
    def __init__(self, u):
        """
        Args:
            u : ManufacturedSolution
                Manufactured solution representing the solution to the diffusion equation.
        """
        super().__init__()
        self.u = u

    def forward(self, X):
        """
        Forward pass to compute the initial condition at locations X.

        Args:
            X : tuple of torch.Tensor
                Input locations where the initial condition is evaluated, all of shape (num_initial_points_i,).

        Returns:
            torch.Tensor
                The computed initial condition at the input locations, shape (num_initial_points_1, num_initial_points_2, ..., num_initial_points_d).
        """
        return self.u.forward(X)
    

class ManufacturedSolutions:
    """
    Container for manufactured solutions and derived functions for the steady diffusion equation.
    """ 
    def __init__(self, N_samples, d, l_min, l_max, b_u_min, b_u_max, c_u_min, c_u_max, b_theta_min, b_theta_max, c_theta_min, c_theta_max, device, dtype, n_RBF):
        """
        Args:
            N_samples (int): Number of samples to generate.
            d (int): Number of spatial dimensions.
            l_min (float): Minimum GRF length scale per dimension.
            l_max (float): Maximum GRF length scale per dimension.
            b_u_min (float): Minimum solution field mean.
            b_u_max (float): Maximum solution field mean.
            c_u_min (float): Minimum solution field standard deviation.
            c_u_max (float): Maximum solution field standard deviation.
            device (str): Device to perform computations on.
            dtype (torch.dtype): Data type for computations.
            n_RBF (int): Number of basis functions per dimension for GRF interpolation.
        """
        super().__init__()
        self.u = ManufacturedSolution(N_samples, # Number of samples
                                 d, # Number of spatial dimensions
                                 l_min, # Minimum GRF length scale per dimension
                                 l_max, # Maximum GRF length scale per dimension
                                 b_u_min, # minimum solution field mean
                                 b_u_max, # maximum solution field mean
                                 c_u_min, # minimum solution field standard deviation
                                 c_u_max, # maximum solution field standard deviation
                                 device, # Device to perform computations on
                                 dtype, # Data type for computations
                                 n_RBF) # Number of basis functions per dimension for GRF interpolation
        self.theta = ManufacturedSolution(N_samples, # Number of samples
                            d, # Number of spatial dimensions
                            l_min, # Minimum GRF length scale per dimension
                            l_max, # Maximum GRF length scale per dimension
                            b_theta_min, # minimum coefficient field mean
                            b_theta_max, # maximum coefficient field mean
                            c_theta_min, # minimum coefficient field standard deviation
                            c_theta_max, # maximum coefficient field standard deviation
                            device, # Device to perform computations on
                            dtype, # Data type for computations
                            n_RBF) # Number of basis functions per dimension for data generation
        self.f = Forcing(self.theta, self.u)
        n_y0 = torch.tensor([0,-1], dtype=dtype, device=device) #Normal vector at y0 boundary
        n_yL = torch.tensor([0,1], dtype=dtype, device=device) #Normal vector at yL boundary
        self.eta_y0, self.eta_yL = NeumannBC(n_y0, self.theta, self.u), NeumannBC(n_yL, self.theta, self.u)
        self.g_x0, self.g_xL = DirichletBC(self.theta, self.u), DirichletBC(self.theta, self.u)
        self.u0 = IC(self.u)