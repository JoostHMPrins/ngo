# Copyright 2025 Joost Prins

# Standard
from itertools import product

# 3rd Party
import numpy as np
import matplotlib.pyplot as plt


class GaussLegendreQuadrature:
    """
    A class to compute Gauss-Legendre quadrature points and weights.

    This class supports multi-dimensional quadrature grids by combining 1D Gauss-Legendre quadrature
    points and weights for each dimension.

    Attributes:
        Q (np.ndarray): Total number of quadrature points per dimension.
        n_elements (np.ndarray): Number of elements per dimension.
        Q_per_element (np.ndarray): Number of quadrature points per element per dimension.
        d (int): Dimensionality of the quadrature grid.
        L_elements (np.ndarray): Length of each element per dimension.
        w (np.ndarray): Quadrature weights. Shape: (n_points,).
        xi (np.ndarray): Quadrature points. Shape: (n_points, d).
    """

    def __init__(self, Q, n_elements):
        """
        Initialize the Gauss-Legendre quadrature.

        Args:
            Q (list or np.ndarray): Total number of quadrature points per dimension.
            n_elements (list or np.ndarray): Number of elements per dimension.
        """
        self.Q = np.array(Q)
        self.n_elements = np.array(n_elements)
        self.Q_per_element = np.array(self.Q/self.n_elements, dtype=int)
        self.d = len(self.Q)
        self.L_elements = 1/np.array(n_elements)
        self.w, self.xi = self.compute_quadrature()

    def compute_quadrature(self):
        """
        Compute the Gauss-Legendre quadrature points and weights.

        Returns:
            tuple:
                - w (np.ndarray): Quadrature weights. Shape: (n_points,).
                - xi (np.ndarray): Quadrature points. Shape: (n_points, d).
        """
        quadrature_data = [np.polynomial.legendre.leggauss(n_points) for n_points in self.Q_per_element]
        rescaled_points = [0.5 * (x + 1) for x, _ in quadrature_data]
        rescaled_points = [rescaled_points[dim] * self.L_elements[dim] for dim in range(self.d)]
        rescaled_weights = [0.5 * w for _, w in quadrature_data]
        rescaled_weights = [rescaled_weights[dim] * self.L_elements[dim] for dim in range(self.d)]
        allpoints = rescaled_points[:]
        allweights = rescaled_weights[:]
        for dim in range(self.d):
            for i in range(1, self.n_elements[dim]):
                newpoints = rescaled_points[dim] + self.L_elements[dim] * i
                allpoints[dim] = np.append(allpoints[dim], newpoints)
                allweights[dim] = np.append(allweights[dim], rescaled_weights[dim])
        xis = np.array(list(product(*allpoints)))
        xis = xis.reshape(-1, xis.shape[-1])
        ws = np.array([np.prod(w_comb) for w_comb in product(*allweights)])
        return ws.flatten(), xis
    
    def plot_2dquadraturegrid(self):
        """
        Plot the 2D quadrature grid with weights as color.
        """
        fig, ax = plt.subplots(1,1, figsize=(6, 4))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
        sc = ax.scatter(self.xi[:,0], self.xi[:,1], c=self.w, s=50)
        plt.axis('square')
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    
    def plot_3dquadraturegrid(self):
        """
        Plot the 3D quadrature grid with weights as color.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot with weights determining the color
        sc = ax.scatter(quad.xi[:, 0], self.xi[:, 1], self.xi[:, 2], c=self.w, cmap='viridis', s=50)
        # Add colorbar and labels
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label('Quadrature Weights')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Scatter Plot of Quadrature Points')
        plt.show()


class UniformQuadrature:
    """
    A class to compute uniform quadrature points and weights.

    Attributes:
        Q (np.ndarray): Number of quadrature points per dimension.
        d (int): Dimensionality of the quadrature grid.
        w (np.ndarray): Quadrature weights. Shape: (n_points,).
        xi (np.ndarray): Quadrature points. Shape: (n_points, d).
    """

    def __init__(self, Q):
        """
        Initialize the uniform quadrature.

        Args:
            Q (list or np.ndarray): Number of quadrature points per dimension.
        """
        self.Q = np.array(Q)
        self.d = len(self.Q)
        self.w, self.xi = self.compute_quadrature()

    def compute_quadrature(self):
        """
        Compute the uniform quadrature points and weights.

        Returns:
            tuple:
                - w (np.ndarray): Quadrature weights. Shape: (n_points,).
                - xi (np.ndarray): Quadrature points. Shape: (n_points, d).
        """
        grid = np.mgrid[[slice(0, 1, q * 1j) for q in self.Q]]
        xis = np.vstack(map(np.ravel, grid)).T
        ws = 1/np.prod(self.Q)*np.ones((np.prod(self.Q)))
        return ws, xis
    
    def plot_2dquadraturegrid(self):
        """
        Plot the 2D quadrature grid with weights as color.
        """
        fig, ax = plt.subplots(1,1, figsize=(6, 4))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
        sc = ax.scatter(self.xi[:,0], self.xi[:,1], c=self.w, s=50)
        plt.axis('square')
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    
    def plot_3dquadraturegrid(self):
        """
        Plot the 3D quadrature grid with weights as color.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot with weights determining the color
        sc = ax.scatter(self.xi[:, 0], self.xi[:, 1], self.xi[:, 2], c=self.w, cmap='viridis', s=50)
        # Add colorbar and labels
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label('Quadrature Weights')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Scatter Plot of Quadrature Points')
        plt.show()