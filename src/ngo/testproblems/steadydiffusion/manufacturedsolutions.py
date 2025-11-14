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
                Input locations where the forcing term is evaluated, all of shape (num_points_i,).

        Returns:
            torch.Tensor
                The computed forcing term at the input locations, shape (num_points_1, num_points_2, ..., num_points_d).
        """
        return -self.theta.forward(X)*torch.sum(self.u.laplacian(X), axis=-1) - torch.sum(self.theta.grad(X)*self.u.grad(X), axis=-1)
    

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
        return opt_einsum.contract('i,N...,N...i->N...', self.n, self.theta.forward(X), self.u.grad(X))
    

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