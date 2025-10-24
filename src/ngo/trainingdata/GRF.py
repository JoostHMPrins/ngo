# # Copyright 2025 Joost Prins

# # 3rd Party
# import numpy as np
# import scipy.spatial as spsp
# import opt_einsum


# class GRF:
#     """
#     Class to generate Gaussian Random Fields (GRFs).

#     Attributes:
#         N_samples (int): Number of GRF samples.
#         l (np.ndarray): Length scale of GRF, shape (d,).
#         d (int): Dimensionality of GRF.
#         n_mus (int): Number of GRF sample locations per volume element 1/l^d.
#         mus (np.ndarray): Sample locations, shape (n_mus, d).
#         f_hat (np.ndarray): RBF interpolation coefficients, shape (N_samples, n_mus).
#     """

#     def __init__(self, N_samples, l):
#         """
#         Initialize the GRF class.

#         Args:
#             N_samples (int): Number of GRF samples.
#             l (list or np.ndarray): Length scale of GRF, shape (d,).
#         """
#         super().__init__()
#         self.N_samples = N_samples 
#         self.l = np.array(l)
#         self.d = len(l)
#         # self.n_mus = int(max(10,1/np.prod(self.l)))
#         self.n_mus = int(max(10,1/np.prod(self.l)))
#         self.mus = self.compute_mus()
#         self.f_hat = self.compute_RBFintcoeffs()
#         self.mus = np.array(self.mus)

#     def compute_mus(self):
#         """
#         Sample random points on the unit square.

#         Returns:
#             np.ndarray: Sample locations, shape (n_mus, d).
#         """
#         mus = np.random.uniform(0,1,size=(self.n_mus,self.d))
#         return mus

#     def compute_cov(self):
#         """
#         Compute Gaussian covariance matrix between points.

#         Returns:
#             np.ndarray: Covariance matrix, shape (n_mus, n_mus).
#         """
#         l = self.l
#         cov = np.exp(-1/2*spsp.distance_matrix(self.mus/l[None,:], self.mus/l[None,:], p=2)**2)
#         return cov
    
#     def compute_GRFpoints(self, cov):
#         """
#         Sample from a multivariate Gaussian with covariance cov.

#         Args:
#             cov (np.ndarray): Covariance matrix, shape (n_mus, n_mus).

#         Returns:
#             np.ndarray: Sampled GRF points, shape (N_samples, n_mus).
#         """
#         f = np.random.multivariate_normal(np.zeros(self.n_mus), cov=cov, size=self.N_samples)
#         return f
    
#     def compute_RBFintcoeffs(self):
#         """
#         Interpolate the GRF with Gaussian RBFs.

#         Returns:
#             np.ndarray: RBF interpolation coefficients, shape (N_samples, n_mus).
#         """
#         cov = self.compute_cov()
#         f = self.compute_GRFpoints(cov)
#         cov = np.array(cov)
#         cov_inv = np.linalg.inv(cov)
#         f = np.array(f)
#         f_hat = opt_einsum.contract('ij,nj->ni', cov_inv, f)
#         return f_hat
    
#     def phi_n(self, i, x):
#         """
#         Compute the RBF values at points x.

#         Args:
#             i (int): Index of the sample.
#             x (np.ndarray): Input points, shape (N_points, d).

#         Returns:
#             np.ndarray: RBF values, shape (N_points, n_mus).
#         """
#         output = self.f_hat[i,None,:]*np.exp(-1/2*np.sum(((x[:,None,:] - self.mus[None,:,:])/self.l[None,None,:])**2, axis=-1))
#         return output
    
#     def forward(self, i):
#         """
#         Forward evaluation of RBF interpolated GRF.

#         Args:
#             i (int): Index of the sample.

#         Returns:
#             function: A function that computes the GRF at a given point x.
#         """
#         def function(x):
#             """ 
#             Args:
#                 x (np.ndarray): Input points, shape (N_points, d).

#             Returns:
#                 np.ndarray: GRF values, shape (N_points,).
#             """
#             phi_n = self.phi_n(i, x)
#             return np.sum(phi_n, axis=1)
#         return function
    
#     def grad(self, i):
#         """
#         Gradient of RBF interpolated GRF.

#         Args:
#             i (int): Index of the sample.

#         Returns:
#             function: A function that computes the gradient of the GRF at a given point x.
#         """
#         def function(x):
#             """
#             Args:
#                 x (np.ndarray): Input points, shape (N_points, d).

#             Returns:
#                 np.ndarray: Gradient of the GRF, shape (N_points, d).
#             """
#             phi_n = self.phi_n(i, x)
#             prefactor = -1/(self.l[None,None,:]**2)*(x[:,None,:] - self.mus[None,:,:])
#             return np.sum(prefactor*phi_n[:,:,None], axis=1)
#         return function

#     def d2dxi2(self, i):
#         """
#         Laplacian of RBF interpolated GRF.

#         Args:
#             i (int): Index of the sample.

#         Returns:
#             function: A function that computes the Laplacian of the GRF at a given point x.
#         """
#         def function(x):
#             """
#             Args:
#                 x (np.ndarray): Input points, shape (N_points, d).

#             Returns:
#                 np.ndarray: Laplacian of the GRF, shape (N_points,).
#             """
#             phi_n = self.phi_n(i, x)
#             prefactor = (x[:,None,:] - self.mus[None,:,:])**2/self.l[None,None,:]**4 - 1/self.l[None,None,:]**2
#             return np.sum(prefactor*phi_n[:,:,None], axis=1)
#         return function
        

# class ScaledGRF:
#     """
#     Class to generate scaled and translated Gaussian Random Fields (GRFs).

#     Attributes:
#         grf (GRF): GRF object.
#         c (np.ndarray): Scaling factors, shape (N_samples,).
#         b (np.ndarray): Translation/offset factors, shape (N_samples,).
#     """

#     def __init__(self, N_samples, l, c, b):
#         """
#         Initialize the ScaledGRF class.

#         Args:
#             N_samples (int): Number of GRF samples.
#             l (list or np.ndarray): Length scale of GRF, shape (d,).
#             c (np.ndarray): Scaling factors, shape (N_samples,).
#             b (np.ndarray): Translation factors, shape (N_samples,).
#         """
#         super().__init__()
#         self.grf = GRF(N_samples, l)
#         self.c = c
#         self.b = b
    
#     def forward(self, i):
#         """
#         Forward evaluation of scaled GRF.

#         Args:
#         i (int): Index of the sample.

#         Returns:
#         function: A function that computes the scaled GRF at a given point x.
#         """
#         def function(x):
#             """
#             Args:
#             x (np.ndarray): Input points, shape (N_points, d).

#             Returns:
#             np.ndarray: Scaled GRF values, shape (N_points,).
#             """
#             return self.c[i]*self.grf.forward(i)(x) + self.b[i]
#         return function
    
#     def grad(self, i):
#         """
#         Gradient of scaled GRF.

#         Args:
#             i (int): Index of the sample.

#         Returns:
#             function: A function that computes the gradient of the scaled GRF at a given point x.
#         """
#         def function(x):
#             """
#             Args:
#                 x (np.ndarray): Input points, shape (N_points, d).

#             Returns:
#                 np.ndarray: Gradient of the scaled GRF, shape (N_points, d).
#             """
#             return self.c[i]*self.grf.grad(i)(x)
#         return function
    
#     def d2dxi2(self, i):
#         """
#         Laplacian of scaled GRF.

#         Args:
#             i (int): Index of the sample.

#         Returns:
#             function: A function that computes the Laplacian of the scaled GRF at a given point x.
#         """
#         def function(x):
#             """
#             Args:
#                 x (np.ndarray): Input points, shape (N_points, d).

#             Returns:
#                 np.ndarray: Laplacian of the scaled GRF, shape (N_points,).
#             """
#             return self.c[i]*self.grf.d2dxi2(i)(x)
#         return function


import numpy as np
import scipy
import opt_einsum


class GRF:
    def __init__(self, N_samples, l):
        super().__init__()
        self.N_samples = N_samples #Number of GRF samples
        self.l = np.array(l) #Length scale of GRF
        self.d = len(l) #Dimensionality of GRF
        self.n_mus = int(np.ceil(1/np.prod(self.l))) #Number of GRF sample locations
        self.mus = self.compute_mus()
        self.f_hat = self.compute_coeffs()

    #Sample n_mus random points on the unit square
    def compute_mus(self):
        mus = np.random.uniform(0,1,size=(self.N_samples,self.n_mus,self.d))
        return mus

    #Compute Gaussian covariance matrix cov between points
    def compute_cov(self):
        scaled_mus = self.mus / self.l[None, None, :]
        diff = scaled_mus[:, None, :, :] - scaled_mus[:, :, None, :]
        dist_sq = np.sum(diff**2, axis=-1)
        return np.exp(-0.5 * dist_sq)
    
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

    # Laplacian of RBF interpolated GRF
    def d2dxi2(self, i):
        def function(x):
            phi_n = self.phi_n(i, x)
            prefactor = (x[:,None,:] - self.mus[i,:,:])**2/self.l[None,None,:]**4 - 1/self.l[None,None,:]**2
            return np.sum(prefactor*phi_n[:,:,None], axis=1)
        return function


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
    def forward_x(self, i, x):        
        return self.c[i]*self.grf.forward_x(i,x) + self.b[i]
        
    
    def forward_nutils(self, i):
        def function(x):
            return self.c[i]*self.grf.forward_nutils(i)(x) + self.b[i]
        return function
    
    def grad(self, i):
        def function(x):
            return self.c[i]*self.grf.grad(i)(x)
        return function
    
    def grad_x(self, i , x ):
        return self.c[i]*self.grf.grad_x(i,x)
    
    def d2dxi2(self, i):
        def function(x):
            return self.c[i]*self.grf.d2dxi2(i)(x)
        return function

    def d2dxi2_x(self, i,x):
        return self.c[i]*self.grf.d2dxi2_x(i,x)