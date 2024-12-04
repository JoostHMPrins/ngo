import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class UniformQuadrature2D:
    def __init__(self, Q):
        self.d = 2
        self.Q = Q
        self.compute_interiorquadrature()
        self.compute_boundaryquadrature()
        
    def compute_interiorquadrature(self):
        x_0_Q, x_1_Q = np.mgrid[0:1:self.Q*1j, 0:1:self.Q*1j]
        self.xi_Omega = np.vstack([x_0_Q.ravel(), x_1_Q.ravel()]).T
        self.w_Omega = 1/(self.Q**self.d)*np.ones((self.Q**self.d))
        
    def compute_boundaryquadrature(self):
        xi_Gamma_i = np.linspace(0,1, self.Q)
        self.xi_Gamma_b = np.zeros((self.Q,self.d))
        self.xi_Gamma_b[:,0] = xi_Gamma_i
        self.xi_Gamma_t = np.ones((self.Q,self.d))
        self.xi_Gamma_t[:,0] = xi_Gamma_i
        self.xi_Gamma_l = np.zeros((self.Q,self.d))
        self.xi_Gamma_l[:,1] = xi_Gamma_i
        self.xi_Gamma_r = np.ones((self.Q,self.d))
        self.xi_Gamma_r[:,1] = xi_Gamma_i
        self.xi_Gamma_eta = np.zeros((2*self.Q,self.d))
        self.xi_Gamma_eta[:self.Q] = self.xi_Gamma_b
        self.xi_Gamma_eta[self.Q:] = self.xi_Gamma_t        
        self.xi_Gamma_g = np.zeros((2*self.Q,self.d))
        self.xi_Gamma_g[:self.Q] = self.xi_Gamma_l
        self.xi_Gamma_g[self.Q:] = self.xi_Gamma_r        
        self.w_Gamma_b = 1/(self.Q)*np.ones((self.Q))
        self.w_Gamma_t = 1/(self.Q)*np.ones((self.Q))
        self.w_Gamma_l = 1/(self.Q)*np.ones((self.Q))
        self.w_Gamma_r = 1/(self.Q)*np.ones((self.Q))
        self.w_Gamma_eta = 1/(self.Q)*np.ones((2*self.Q))
        self.w_Gamma_g = 1/(self.Q)*np.ones((2*self.Q))
        self.xi_Gamma = np.zeros((4,self.Q,2))
        self.xi_Gamma[0] = self.xi_Gamma_l
        self.xi_Gamma[1] = self.xi_Gamma_r
        self.xi_Gamma[2] = self.xi_Gamma_b
        self.xi_Gamma[3] = self.xi_Gamma_t
        self.xi_Gamma = self.xi_Gamma.reshape(self.xi_Gamma.shape[0]*self.xi_Gamma.shape[1],self.xi_Gamma.shape[2])
        self.w_Gamma = np.array([self.w_Gamma_l,self.w_Gamma_r,self.w_Gamma_b,self.w_Gamma_t]).flatten()
        
    def plot_quadraturegrid(self):
        plt.scatter(self.xi_Omega[:,0], self.xi_Omega[:,1], c=self.w_Omega, s=10)
        plt.scatter(self.xi_Gamma[:,0], self.xi_Gamma[:,1], c=self.w_Gamma, s=10)
        plt.axis('square')
        plt.colorbar()
        # plt.savefig("U_Q"+str(self.Q)+".svg", bbox_inches='tight')#, transparent=True)


import numpy as np
from itertools import product

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
        # plt.savefig("GL_Q"+str(self.Q)+".svg", bbox_inches='tight')#, transparent=True)
    
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
        # plt.savefig("ndquadrature.svg", bbox_inches='tight')#, transparent=True)
        plt.show()
            
        
class GaussLegendreQuadrature2D:
    def __init__(self, Q, n_elements):
        self.d = 2
        self.Q = Q
        self.n_elements = n_elements
        self.L_element = 1/n_elements
        self.compute_interiorquadrature()
        self.compute_boundaryquadrature()
        
    def compute_interiorquadrature(self):
        x, w = np.polynomial.legendre.leggauss(int(self.Q/self.n_elements))
        x = np.array(np.meshgrid(x,x,indexing='ij')).reshape(2,-1).T/2 + 0.5
        w = w/2*self.L_element
        w = (w*w[:,None]).ravel()
        xi_Omega = []
        w_Omega = []
        for i in range(self.n_elements):
            for j in range(self.n_elements):
                newpts = self.L_element*x
                newpts[:,0] = newpts[:,0] + self.L_element*i
                newpts[:,1] = newpts[:,1] + self.L_element*j
                xi_Omega.append(newpts)
                w_Omega.append(w)
        self.w_Omega = np.array(w_Omega).flatten()
        xi_Omega = np.array(xi_Omega)
        self.xi_Omega = xi_Omega.reshape(xi_Omega.shape[0]*xi_Omega.shape[1],xi_Omega.shape[2])

    def compute_boundaryquadrature(self):
        x, w = np.polynomial.legendre.leggauss(int(self.Q/self.n_elements))
        x = x/2 + 0.5
        w = w/2*self.L_element
        xi_Gamma_i = []
        w_Gamma_i = []
        for i in range(self.n_elements):
            xi_Gamma_i.append(self.L_element*x + self.L_element*i)
            w_Gamma_i.append(w)
        xi_Gamma_i = np.array(xi_Gamma_i)
        xi_Gamma_i = xi_Gamma_i.flatten()
        w_Gamma_i = np.array(w_Gamma_i).flatten()
        self.xi_Gamma_b = np.zeros((self.Q,self.d))
        self.xi_Gamma_b[:,0] = xi_Gamma_i
        self.xi_Gamma_t = np.ones((self.Q,self.d))
        self.xi_Gamma_t[:,0] = xi_Gamma_i
        self.xi_Gamma_l = np.zeros((self.Q,self.d))
        self.xi_Gamma_l[:,1] = xi_Gamma_i
        self.xi_Gamma_r = np.ones((self.Q,self.d))
        self.xi_Gamma_r[:,1] = xi_Gamma_i
        self.xi_Gamma_eta = np.zeros((2*self.Q,self.d))
        self.xi_Gamma_eta[:self.Q] = self.xi_Gamma_b
        self.xi_Gamma_eta[self.Q:] = self.xi_Gamma_t        
        self.xi_Gamma_g = np.zeros((2*self.Q,self.d))
        self.xi_Gamma_g[:self.Q] = self.xi_Gamma_l
        self.xi_Gamma_g[self.Q:] = self.xi_Gamma_r 
        self.w_Gamma_b = w_Gamma_i
        self.w_Gamma_t = w_Gamma_i
        self.w_Gamma_l = w_Gamma_i
        self.w_Gamma_r = w_Gamma_i
        self.w_Gamma_eta = np.zeros((2*len(w_Gamma_i)))
        self.w_Gamma_eta[:len(w_Gamma_i)] = self.w_Gamma_b
        self.w_Gamma_eta[len(w_Gamma_i):] = self.w_Gamma_t
        self.w_Gamma_g = np.zeros((2*len(w_Gamma_i)))
        self.w_Gamma_g[:len(w_Gamma_i)] = self.w_Gamma_l
        self.w_Gamma_g[len(w_Gamma_i):] = self.w_Gamma_r
        self.xi_Gamma = np.zeros((4,self.Q,2))
        self.xi_Gamma[0] = self.xi_Gamma_l
        self.xi_Gamma[1] = self.xi_Gamma_r
        self.xi_Gamma[2] = self.xi_Gamma_b
        self.xi_Gamma[3] = self.xi_Gamma_t
        self.xi_Gamma = self.xi_Gamma.reshape(self.xi_Gamma.shape[0]*self.xi_Gamma.shape[1],self.xi_Gamma.shape[2])
        self.w_Gamma = np.array([w_Gamma_i,w_Gamma_i,w_Gamma_i,w_Gamma_i]).flatten()

    def plot_quadraturegrid(self):
        plt.scatter(self.xi_Omega[:,0], self.xi_Omega[:,1], c=self.w_Omega, s=1)
        plt.scatter(self.xi_Gamma[:,0], self.xi_Gamma[:,1], c=self.w_Gamma, s=1)
        plt.axis('square')
        plt.colorbar()
        # plt.savefig("GL_Q"+str(self.Q)+".svg", bbox_inches='tight')#, transparent=True)

          
class UnitSquareOutwardNormal:
    def __init__(self, Q):
        self.d = 2
        self.Q = Q
        self.compute_outwardnormal()
        
    def compute_outwardnormal(self):
        n_b = np.array([0,-1])
        self.n_b = np.tile(n_b,(self.Q,1))
        n_t = np.array([0,1])
        self.n_t = np.tile(n_t,(self.Q,1))
        n_l = np.array([-1,0])
        self.n_l = np.tile(n_l,(self.Q,1))
        n_r = np.array([1,0])
        self.n_r = np.tile(n_r,(self.Q,1))
        self.n_Gamma_eta = np.zeros((2*self.Q,self.d))
        self.n_Gamma_eta[:self.Q] = self.n_b
        self.n_Gamma_eta[self.Q:] = self.n_t
        self.n_Gamma_g = np.zeros((2*self.Q,self.d))
        self.n_Gamma_g[:self.Q] = self.n_l
        self.n_Gamma_g[self.Q:] = self.n_r


class UnitSquareOutwardNormal2D:
    def __init__(self, Q):
        self.d = 2
        self.Q = Q
        self.compute_outwardnormal()
        
    def compute_outwardnormal(self):
        n_b = np.array([0,-1])
        self.n_b = np.tile(n_b,(self.Q[0],1))
        n_t = np.array([0,1])
        self.n_t = np.tile(n_t,(self.Q[0],1))
        n_l = np.array([-1,0])
        self.n_l = np.tile(n_l,(self.Q[1],1))
        n_r = np.array([1,0])
        self.n_r = np.tile(n_r,(self.Q[1],1))
        self.n_Gamma_eta = np.zeros((2*self.Q[0],self.d))
        self.n_Gamma_eta[:self.Q[0]] = self.n_b
        self.n_Gamma_eta[self.Q[0]:] = self.n_t
        self.n_Gamma_g = np.zeros((2*self.Q[1],self.d))
        self.n_Gamma_g[:self.Q[1]] = self.n_l
        self.n_Gamma_g[self.Q[1]:] = self.n_r