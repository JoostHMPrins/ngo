# Copyright 2025 Joost Prins

# Standard
from itertools import product

# 3rd Party
import numpy as np
import matplotlib.pyplot as plt


class GaussLegendreQuadrature:
    def __init__(self, Q, n_elements):
        self.Q = np.array(Q)
        self.n_elements = np.array(n_elements)
        self.Q_per_element = np.array(self.Q/self.n_elements, dtype=int)
        self.d = len(self.Q)
        self.L_elements = 1/np.array(n_elements)
        self.w, self.xi = self.compute_quadrature()

    def compute_quadrature(self):
        # Get Gauss-Legendre quadrature points and weights for each dimension
        quadrature_data = [np.polynomial.legendre.leggauss(n_points) for n_points in self.Q_per_element]
        # Rescale points and adjust weights for each dimension
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
        fig, ax = plt.subplots(1,1, figsize=(6, 4))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
        sc = ax.scatter(self.xi[:,0], self.xi[:,1], c=self.w, s=50)
        plt.axis('square')
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    
    def plot_3dquadraturegrid(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot with weights determining the color
        sc = ax.scatter(quad.xi[:, 0], self.xi[:, 1], self.xi[:, 2], c=quad.w, cmap='viridis', s=50)
        # Add colorbar and labels
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label('Quadrature Weights')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Scatter Plot of Quadrature Points')
        plt.show()


class UniformQuadrature:
    def __init__(self, Q):
        self.Q = np.array(Q)
        self.d = len(self.Q)
        self.w, self.xi = self.compute_quadrature()

    def compute_quadrature(self):
        grid = np.mgrid[[slice(0, 1, q * 1j) for q in self.Q]]
        xis = np.vstack(map(np.ravel, grid)).T
        ws = 1/np.prod(self.Q)*np.ones((np.prod(self.Q)))
        return ws, xis
    
    def plot_2dquadraturegrid(self):
        fig, ax = plt.subplots(1,1, figsize=(6, 4))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
        sc = ax.scatter(self.xi[:,0], self.xi[:,1], c=self.w, s=50)
        plt.axis('square')
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    
    def plot_3dquadraturegrid(self):
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