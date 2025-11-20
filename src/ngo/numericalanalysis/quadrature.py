# Copyright 2025 Joost Prins

# 3rd Party
import numpy as np
import torch


class GaussLegendreQuadrature:
    """
    Class to compute Gauss-Legendre quadrature points and weights over a domain
    divided into multiple elements.
    """
    def __init__(self, Q, n_elements, dtype, device):
        """
        Initialize the Gauss-Legendre quadrature.

        Args:
            dtype: Data type for the tensors.
            device: Device for the tensors.
            Q (int): Total number of quadrature points.
            n_elements (int): Number of elements to divide the domain into.
        
        Returns:
            None
        """
        self.dtype = dtype
        self.device = device
        self.Q = Q
        self.n_elements = n_elements
        # Compute number of quadrature points per element and element length
        self.Q_per_element = Q//n_elements
        # Length of a single element, assuming domain length is 1
        self.L_element = 1/n_elements
        # Compute quadrature points and weights
        self.w, self.xi = self.compute_quadrature()

    def compute_quadrature(self):
        """
        Compute the Gauss-Legendre quadrature points and weights over the entire domain.

        Args:
            None
            
        Returns:
            torch.Tensor: Weights of shape (Q,).
            torch.Tensor: Quadrature points of shape (Q,).
        """
        # Compute 1D Gauss-Legendre quadrature points and weights for a single element in [-1,1]
        xi, w = np.polynomial.legendre.leggauss(self.Q_per_element)
        # Rescale points and weights to [0, 1]
        xi = 0.5*(xi + 1)
        w = 0.5*w
        # Rescale points and weights to element length
        xi_single = xi*self.L_element
        w_single = w*self.L_element
        # Initialize arrays to hold all quadrature points and weights
        xi = np.empty(0)
        w = np.empty(0)
        # Loop over elements to create full set of quadrature points and weights
        for i in range(0, self.n_elements):
            newpoints = xi_single + self.L_element * i
            xi = np.append(xi, newpoints)
            w = np.append(w, w_single)
        w = torch.tensor(w, dtype=self.dtype, device=self.device)
        xi = torch.tensor(xi, dtype=self.dtype, device=self.device)
        return w, xi
    

class UniformQuadrature:
    """
    Class to compute uniform quadrature points and weights over [0, 1].
    """
    def __init__(self, Q, dtype, device):
        """
        Initialize the Uniform quadrature.

        Args:
            Q (int): Total number of quadrature points.
            dtype: Data type for the tensors.
            device: Device for the tensors.

        Returns:
            None
        """
        self.dtype = dtype
        self.device = device
        self.Q = Q
        self.w, self.xi = self.compute_quadrature()

    def compute_quadrature(self):
        """
        Compute the uniform quadrature points and weights over [0, 1].

        Args:
            None

        Returns:
            torch.Tensor: Weights of shape (Q,).
            torch.Tensor: Quadrature points of shape (Q,).
        """
        w = np.ones(self.Q)/self.Q
        xi = np.linspace(0, 1, self.Q)
        w = torch.tensor(w, dtype=self.dtype, device=self.device)
        xi = torch.tensor(xi, dtype=self.dtype, device=self.device)
        return w, xi