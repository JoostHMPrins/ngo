import numpy as np
from scipy.spatial import distance_matrix
import opt_einsum
from numba import njit

@njit(parallel=True)
def compute_phi_n(f_hat_i, x, mus_i, l):
    N_points, d = x.shape
    n_mus = mus_i.shape[0]

    # Reshape for broadcasting: (N_points, 1, d)
    x_reshaped = np.reshape(x, (N_points, 1, d))
    mus_reshaped = np.reshape(mus_i, (1, n_mus, d))
    l_reshaped = np.reshape(l, (1, 1, d))

    # Compute squared distances
    diff = (x_reshaped - mus_reshaped) / l_reshaped
    sq_dist = np.sum(diff ** 2, axis=2)

    # Compute RBF values and multiply
    f_hat_i_reshaped = np.reshape(f_hat_i, (1, n_mus))
    return np.exp(-0.5 * sq_dist) * f_hat_i_reshaped


@njit(parallel=True)
def compute_grad_phi_n(f_hat_i, x, mus_i, l):
    N_points, d = x.shape
    n_mus = mus_i.shape[0]

    x_reshaped = np.reshape(x, (N_points, 1, d))         
    mus_reshaped = np.reshape(mus_i, (1, n_mus, d))      
    l_reshaped = np.reshape(l, (1, 1, d))                

    diff = (x_reshaped - mus_reshaped) / l_reshaped      
    sq_dist = np.sum(diff ** 2, axis=2)                  
    phi_n = np.exp(-0.5 * sq_dist) * f_hat_i.reshape(1, n_mus)  # (N_points, n_mus)

    prefactor = -diff / l_reshaped                       

    phi_n_reshaped = phi_n.reshape(N_points, n_mus, 1)  # Instead of phi_n[:, :, None]
    return np.sum(prefactor * phi_n_reshaped, axis=1)




@njit(parallel=True)
def compute_laplacian_phi_n(f_hat_i, x, mus_i, l):
    N_points, d = x.shape
    n_mus = mus_i.shape[0]

    x_reshaped = np.reshape(x, (N_points, 1, d))         # (N, 1, d)
    mus_reshaped = np.reshape(mus_i, (1, n_mus, d))      # (1, n_mus, d)
    l_reshaped = np.reshape(l, (1, 1, d))                # (1, 1, d)

    diff = (x_reshaped - mus_reshaped) / l_reshaped      # (N, n_mus, d)
    sq_dist = np.sum(diff ** 2, axis=2)                  # (N, n_mus)
    phi_n = np.exp(-0.5 * sq_dist) * f_hat_i.reshape(1, n_mus)  # (N, n_mus)

    prefactor = (diff**2 - 1) / (l_reshaped ** 2)        # (N, n_mus, d)
    phi_n_reshaped = phi_n.reshape(N_points, n_mus, 1)  # Instead of phi_n[:, :, None]

    return np.sum(prefactor * phi_n_reshaped, axis=1)  # (N, d)



class GRF:
    def __init__(self, N_samples, l):
        super().__init__()
        self.N_samples = N_samples #Number of GRF samples
        self.l = np.array(l) #Length scale of GRF
        self.d = len(l) #axisensionality of GRF
        self.n_mus = int(np.ceil(1/np.prod(self.l))) #Number of GRF sample locations
        self.mus = self.compute_mus()
        self.f_hat = self.compute_coeffs()

    #Sample n_mus random points on the unit square
    def compute_mus(self):
        mus = np.random.uniform(0,1,size=(self.N_samples,self.n_mus,self.d))
        return mus

    #Compute Gaussian covariance matrix cov between points
    def compute_cov(self):
        scaled_mus = self.mus/self.l[None,None,:]
        scaled_distance_matrix_squared = np.sum((scaled_mus[:,None,:,:] - scaled_mus[:,:, None,:])**2, axis=-1)
        cov = np.exp(-1/2*scaled_distance_matrix_squared)
        return cov
    
    def compute_coeffs(self):
        cov = self.compute_cov()
        L = np.linalg.cholesky(cov)
        z = np.random.normal(size=(self.N_samples, self.n_mus))
        f = opt_einsum.contract('Nij,Nj->Ni', L, z)
        L_inv = np.linalg.inv(L)
        cov_inv = opt_einsum.contract('Nij,Njk->Nik', L_inv.transpose(0,2,1), L_inv)
        f_hat = opt_einsum.contract('Nij,Nj->Ni', cov_inv, f)
        return f_hat

    def phi_n(self, i, x):
        output = self.f_hat[i,None,:]*np.exp(-1/2*np.sum(((x[:,None,:] - self.mus[i,:,:])/self.l[None,None,:])**2, axis=-1))
        return output

    # def phi_n(self, i, x):
    #     return compute_phi_n(self.f_hat[i], x, self.mus[i], self.l)
    
    #Forward evaluation of RBF interpolated GRF
    def forward(self, i):
        def function(x):
            phi_n = self.phi_n(i, x)
            return np.sum(phi_n, axis=1)
        return function

    #Pointwise forward evaluation of RBF interpolated GRF (required for Nutils)
    def forward_nutils(self, i):
        def function(x):
            phi_n = self.f_hat[i]*np.exp(-np.sum((x - self.mus)**2, axis=-1)/(2*self.l**2))
            return np.sum(phi_n)
        return function
    
    # Gradient of RBF interpolated GRF
    def grad(self, i):
        def function(x):
            phi_n = self.phi_n(i, x)
            prefactor = -1/(self.l[None,None,:]**2)*(x[:,None,:] - self.mus[i,:,:])
            return np.sum(prefactor*phi_n[:,:,None], axis=1)
        return function

    # def grad(self, i):
    #     def function(x):
    #         return compute_grad_phi_n(self.f_hat[i], x, self.mus[i], self.l)
    #     return function

    # Laplacian of RBF interpolated GRF
    def d2dxi2(self, i):
        def function(x):
            phi_n = self.phi_n(i, x)
            prefactor = (x[:,None,:] - self.mus[i,:,:])**2/self.l[None,None,:]**4 - 1/self.l[None,None,:]**2
            return np.sum(prefactor*phi_n[:,:,None], axis=1)
        return function

    # def d2dxi2(self, i):
    #     def function(x):
    #         return compute_laplacian_phi_n(self.f_hat[i], x, self.mus[i], self.l)
    #     return function
        

class ScaledGRF:
    def __init__(self, N_samples, l, c, b):
        super().__init__()
        self.grf = GRF(N_samples, l)
        #Scaling and translation of GRF f'(x) = c f(x) + b
        self.c = c
        self.b = b
    
    def forward(self, i):
        def function(x):
            return self.c[i]*self.grf.forward(i)(x) + self.b[i]
        return function
    
    def forward_nutils(self, i):
        def function(x):
            return self.c[i]*self.grf.forward_nutils(i)(x) + self.b[i]
        return function
    
    def grad(self, i):
        def function(x):
            return self.c[i]*self.grf.grad(i)(x)
        return function
    
    def d2dxi2(self, i):
        def function(x):
            return self.c[i]*self.grf.d2dxi2(i)(x)
        return function