import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from joblib import Parallel, delayed
import opt_einsum


# def conserve_mass_and_energy(model, d_x, u_0_m, C_m, C, u_m_hat):


#     d_x = d_x#*model.hparams['Dt']
#     C_m = C_m#*model.hparams['Dt']
#     C = C#*model.hparams['Dt']
#     M = model.M_rr#*model.hparams['Dt']

#     CTC = opt_einsum.contract('Nn,Nn->N', C_m, C_m)
#     CTd = opt_einsum.contract('Nn,Nn->N', C_m, d_x)
#     uTd = opt_einsum.contract('Nn,Nn->N', u_m_hat, d_x)
#     C_hat = opt_einsum.contract('Nn,Nn->N', u_m_hat, C_m)
#     u_im1TMu_im1 = opt_einsum.contract('Nn,mn,Nn->N', u_0_m, M, u_0_m)

#     alpha = model.hparams['alpha']

#     Q = 1/2*M + alpha*model.Identity/model.hparams['Dt']

#     uTQu = opt_einsum.contract('Nm,mn,Nn->N', u_m_hat, Q, u_m_hat)
#     uTQC = opt_einsum.contract('Nm,mn,Nn->N', u_m_hat, Q, C_m)
#     CTQu = opt_einsum.contract('Nm,mn,Nn->N', C_m, Q, u_m_hat)
#     CTQC = opt_einsum.contract('Nn,mn,Nn->N', C_m, Q, C_m)
    
#     non_dissipative = uTQu > 1/2*u_im1TMu_im1 + uTd

#     A = opt_einsum.contract('Nm,mn,Nn->N', u_m_hat - (C_hat/CTC)[:,None]*C_m, Q, u_m_hat - (C_hat/CTC)[:,None]*C_m)
#     B = C/CTC*(uTQC + CTQu) - 2*C*C_hat/CTC**2*CTQC - uTd + C_hat/CTC*CTd
#     D = (C/CTC)**2*CTQC - C/CTC*CTd - 1/2*u_im1TMu_im1

#     a1 = 1/torch.clamp(2*A, min=1e-7)*(-B + torch.sqrt(B**2 - 4*A*D))
#     a2 = 1/torch.clamp(2*A, min=1e-7)*(-B - torch.sqrt(B**2 - 4*A*D))

#     a1[a1<0]=torch.inf
#     a2[a2<0]=torch.inf
#     a_plus = torch.min(a1,a2)

#     a = torch.ones(u_m_hat.shape[0], dtype=model.dtype, device=model.device)
#     a[non_dissipative] = a_plus[non_dissipative]

#     b = C/CTC - C_hat/CTC*a

#     u_m_hat = a[:,None]*u_m_hat + b[:,None]*C_m

#     return u_m_hat


def conserve_mass_and_energy(model, d_x, u_0_m, C_m, C, u_m_hat):

    d_x = d_x#*model.hparams['Dt']
    C_m = C_m#*model.hparams['Dt']
    C = C#*model.hparams['Dt']
    M = model.M_rr#*model.hparams['Dt']

    CTC = opt_einsum.contract('Nn,Nn->N', C_m, C_m)
    CTd = opt_einsum.contract('Nn,Nn->N', C_m, d_x)
    uTd = opt_einsum.contract('Nn,Nn->N', u_m_hat, d_x)
    C_hat = opt_einsum.contract('Nn,Nn->N', u_m_hat, C_m)
    u_im1TMu_im1 = opt_einsum.contract('Nn,mn,Nn->N', u_0_m, M, u_0_m)

    alpha =  model.hparams['alpha']

    Q = 1/2*M + alpha*model.Identity/model.hparams['Dt']

    uTQu = opt_einsum.contract('Nm,mn,Nn->N', u_m_hat, Q, u_m_hat)
    uTQC = opt_einsum.contract('Nm,mn,Nn->N', u_m_hat, Q, C_m)
    CTQu = opt_einsum.contract('Nm,mn,Nn->N', C_m, Q, u_m_hat)
    CTQC = opt_einsum.contract('Nn,mn,Nn->N', C_m, Q, C_m)
    
    non_dissipative = uTQu > 1/2*u_im1TMu_im1 + uTd

    A = opt_einsum.contract('Nm,mn,Nn->N', u_m_hat - (C_hat/CTC)[:,None]*C_m, Q, u_m_hat - (C_hat/CTC)[:,None]*C_m)
    B = C/CTC*(uTQC + CTQu) - 2*C*C_hat/CTC**2*CTQC - uTd + C_hat/CTC*CTd
    D = (C/CTC)**2*CTQC - C/CTC*CTd - 1/2*u_im1TMu_im1

    # BdivA = B/torch.clamp(A, min=1e-7)
    # ADdivB2 = A*D/torch.clamp(B**2, min=1e-7)
    # a1 = -BdivA/2*(1 + torch.sqrt(1 - 4*ADdivB2))
    # a2 = -BdivA/2*(1 - torch.sqrt(1 - 4*ADdivB2))

    a1 = 1/torch.clamp(2*A, min=1e-7)*(-B + torch.sqrt(B**2 - 4*A*D))
    a2 = 1/torch.clamp(2*A, min=1e-7)*(-B - torch.sqrt(B**2 - 4*A*D))

    # mask = torch.abs(a1) < torch.abs(a2)
    # a_plus = torch.where(mask, a1, a2)

    a1[a1<0]=torch.inf
    a2[a2<0]=torch.inf
    a_plus = torch.min(a1,a2)

    a = torch.ones(u_m_hat.shape[0], dtype=model.dtype, device=model.device)
   
    a[non_dissipative] = a_plus[non_dissipative]

    # a[a==torch.inf] = 1

    b = C/CTC - C_hat/CTC*a
    # print(a[non_dissipative])
    # print(b[non_dissipative])

    u_m_hat = a[:,None]*u_m_hat + b[:,None]*C_m

    return u_m_hat


def conserve_energy(model, F, d_x, u_0_m, u_m_hat):

    M = model.M_rr

    uTd = opt_einsum.contract('Nn,Nn->N', u_m_hat, d_x)
    uTMu = opt_einsum.contract('Nn,mn,Nn->N', u_m_hat, M, u_m_hat)
    uTu = opt_einsum.contract('Nn,Nn->N', u_m_hat, u_m_hat)
    u_im1TMu_im1 = opt_einsum.contract('Nn,mn,Nn->N', u_0_m, M, u_0_m)
    
    non_dissipative = 1/2*uTMu - 1/2*u_im1TMu_im1 > -model.hparams['alpha']/model.hparams['Dt']*uTu + uTd

    A = 1/2*uTMu + model.hparams['alpha']/model.hparams['Dt']*uTu
    B = -uTd
    C = -1/2*u_im1TMu_im1

    a1 = 1/torch.clamp(2*A, min=1e-7)*(-B + torch.sqrt(B**2 - 4*A*C))
    a2 = 1/torch.clamp(2*A, min=1e-7)*(-B - torch.sqrt(B**2 - 4*A*C))

    a1[a1<0]=torch.inf
    a2[a2<0]=torch.inf
    a_plus = torch.min(a1,a2)

    a = torch.ones(u_m_hat.shape[0], dtype=model.dtype, device=model.device)
   
    a[non_dissipative] = a_plus[non_dissipative]

    u_m_hat = a[:,None]*u_m_hat

    return u_m_hat


def M_norm_scaling(model, A_hat, u_0_m):

    M_lr = torch.tensor(model.M_lr, dtype=model.dtype, device=model.device)
    M_rr = torch.tensor(model.M_rr, dtype=model.dtype, device=model.device)

    B = A_hat@M_lr
    u_m_hat = opt_einsum.contract('Nmn,Nn->Nm', B, u_0_m)

    uTMu = opt_einsum.contract('Nn,mn,Nn->N', u_m_hat, M_rr, u_m_hat)
    uTu = opt_einsum.contract('Nn,Nn->N', u_m_hat, u_m_hat)
    u_im1TMu_im1 = opt_einsum.contract('Nn,mn,Nn->N', u_0_m, M_rr, u_0_m)
    
    non_dissipative = 1/2*uTMu - 1/2*u_im1TMu_im1 > -model.hparams['alpha']*uTu#/model.hparams['Dt']

    A = 1/2*uTMu + model.hparams['alpha']*uTu#/model.hparams['Dt']
    C = -1/2*u_im1TMu_im1

    a_plus = torch.sqrt(-C)/torch.clamp(torch.sqrt(A), min=1e-7)

    a = torch.ones(u_m_hat.shape[0], dtype=model.dtype, device=model.device)
   
    a[non_dissipative] = a_plus[non_dissipative]

    # print(a)

    A_hat = a[:,None,None]*torch.clone(A_hat)

    return A_hat



class ConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, padding=0, dilation=1):
        """
        Generalized N-dimensional convolution module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.
            padding (int or tuple, optional): Zero-padding added to all sides. Default: 0.
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
            bias (bool, optional): If True, adds a learnable bias to the output. Default: True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Ensure kernel_size, stride, padding, and dilation are tuples matching spatial dimensions
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride if isinstance(stride, tuple) else (stride,) * len(self.kernel_size)
        self.padding = padding if isinstance(padding, tuple) else (padding,) * len(self.kernel_size)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * len(self.kernel_size)

        # Initialize weights and bias
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Forward pass for the generalized N-dimensional convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, *spatial_dims).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, *spatial_dims_out).
        """
        batch_size, in_channels, *spatial_dims = x.shape
        ndim = len(spatial_dims)

        # Adjust padding, stride, and dilation to match the number of dimensions
        assert len(self.kernel_size) == ndim, "Kernel size dimensions must match input dimensions"

        # Pad the input tensor
        pad = [(p, p) for p in reversed(self.padding)]  # Reverse to get correct padding order
        pad = [item for sublist in pad for item in sublist]  # Flatten padding list
        x_padded = torch.nn.functional.pad(x, pad, mode='constant', value=0)

        # Compute output spatial dimensions
        out_dims = [
            (spatial_dims[i] + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) // self.stride[i] + 1
            for i in range(ndim)
        ]

        # Extract patches using unfold (this is done sequentially for each spatial dimension)
        patches = x_padded.unfold(2, self.kernel_size[0], self.stride[0])
        for i in range(1, ndim):
            patches = patches.unfold(2 + i, self.kernel_size[i], self.stride[i])

        # Reshape patches to prepare for convolution
        patches = patches.contiguous().view(batch_size, in_channels, -1, *out_dims)

        # Perform convolution via einsum
        weight_shape = self.weight.shape
        weight = self.weight.view(self.out_channels, self.in_channels, -1)
        out = torch.einsum("bci...,oci->bo...", patches, weight)

        # Add bias if applicable
        if self.bias is not None:
            out += self.bias.view(1, -1, *[1] * ndim)

        return out
    

def balance_num_trainable_params(model, N_w):
    model.free_parameter = 1
    N_w_real_0 = model.hparams['N_w_real']
    free_parameter_values_list = []
    N_w_real = N_w_real_0
    while N_w_real < N_w:
        model.init_layers()
        N_w_real = N_w_real_0 + sum(p.numel() for p in model.parameters())
        free_parameter_values_list.append(model.free_parameter)
        model.free_parameter +=1
    if len(free_parameter_values_list)>1:
        model.free_parameter = free_parameter_values_list[-2]
    else:
        model.free_parameter = free_parameter_values_list[0]
    model.init_layers()
    count = sum(p.numel() for p in model.parameters())
    model.hparams['N_w_real'] += count
    if model.hparams['N_w_real']>N_w:
        raise ValueError('Not enough trainable parameter budget.')
    return model  

# @njit(parallel=True)
def discretize_functions(f_list, x):
    f_discretized = Parallel(n_jobs=-1)(delayed(f)(x) for f in f_list)
    f_discretized = np.array(f_discretized)
    # f_discretized = np.zeros((len(f_list), x.shape[0]))
    # for i in range(len(f_list)):
    #     # print(i)
    #     f_discretized[i] = f_list[i](x)
    return f_discretized


# def rho_max_power_iteration(B, x, tol, max_iterations, dtype, device):
#     """
#     Estimate the spectral radius (largest eigenvalue in magnitude) of a batch of matrices 
#     using the power iteration method.

#     Args:
#     B : ndarray of shape (N, M, M)
#         A batch of N symmetric square matrices of shape (M, M) whose spectral radii are to be estimated.
#     x : ndarray of shape (M,)
#         An initial guess for the eigenvector. This vector is tiled across the batch dimension.
#     tol : float
#         The convergence tolerance. Iteration continues until the maximum relative change in 
#         spectral radius estimates across the batch falls below this threshold.

#     Returns:
#     rho : ndarray of shape (N,)
#         The estimated spectral radii (largest eigenvalues) for each matrix in the batch.
#     """
#     #Add batch dimension to initial random vector
#     x_norm = (torch.sum(x**2))**(1/2)
#     x = x/x_norm
#     x = torch.tile(x,(B.shape[0],1))
#     #Initialize the relative difference and spectral radius estimate arrays
#     reldiff = torch.ones(B.shape[0], dtype=dtype, device=device)
#     rho = torch.ones(B.shape[0], dtype=dtype, device=device)
#     i=0
#     while torch.amax(reldiff)>tol and i<max_iterations:
#         #Compute new spectral radius estimate
#         Bx = opt_einsum.contract('Nij,Nj->Ni', B, x)
#         xTBx = opt_einsum.contract('Ni,Ni->N', x, Bx)
#         xTx = opt_einsum.contract('Ni,Ni->N', x, x)
#         rho_new = xTBx/torch.max(xTx, 1e-7*torch.ones_like(xTx))
#         #Compute absolute relative difference between estimate and previous estimate
#         reldiff = torch.abs((rho_new - rho)/rho)
#         #Update spectral radius estimate
#         print(i)
#         print(torch.sum(reldiff>tol))
#         rho[reldiff>tol] = rho_new[reldiff>tol]
#         #Update eigenvector estimate
#         BxTBx = opt_einsum.contract('Ni,Ni->N', Bx, Bx)
#         x = Bx/torch.max(BxTBx[:,None], 1e-7*torch.ones_like(BxTBx[:,None]))
#         i+=1
#     # print(i)
#     if i==max_iterations:
#         print('Power iteration did not converge')
#         rho = torch.ones(B.shape[0], dtype=dtype, device=device)
#     # print(rho)
#     v = x[0]/(torch.sum(x[0]**2, axis=-1))**(1/2)
#     return rho, v


# def rho_max_power_iteration(B, x, tol, max_iterations, dtype, device):
#     """
#     Estimate the spectral radius (largest eigenvalue in magnitude) of a batch of matrices 
#     using the power iteration method.

#     Args:
#     B : torch.Tensor of shape (N, M, M)
#         A batch of N symmetric square matrices of shape (M, M) whose spectral radii are to be estimated.
#     x : torch.Tensor of shape (M,)
#         An initial guess for the eigenvector. This vector is tiled across the batch dimension.
#     tol : float
#         The convergence tolerance. Iteration continues until the maximum relative change in 
#         spectral radius estimates across the batch falls below this threshold.
#     max_iterations : int
#         Maximum number of iterations to perform.
#     dtype : torch.dtype
#         Data type for tensors.
#     device : torch.device
#         Device on which computations will be performed.

#     Returns:
#     rho : torch.Tensor of shape (N,)
#         The estimated spectral radii (largest eigenvalues) for each matrix in the batch.
#     v : torch.Tensor of shape (M,)
#         The normalized eigenvector estimate for the first matrix in the batch.
#     """
#     # Normalize and expand initial vector
#     x_norm = (torch.sum(x ** 2)) ** 0.5
#     x = x / x_norm
#     x = torch.tile(x, (B.shape[0], 1))

#     # Initialize arrays
#     rho = torch.ones(B.shape[0], dtype=dtype, device=device)
#     not_converged = torch.ones(B.shape[0], dtype=dtype, device=device)

#     i = 0
#     while i < max_iterations and torch.sum(not_converged)>0:
#         # Compute Bx
#         Bx = opt_einsum.contract('Nij,Nj->Ni', B, x)
#         xTBx = opt_einsum.contract('Ni,Ni->N', x, Bx)
#         xTx = opt_einsum.contract('Ni,Ni->N', x, x)

#         # Spectral radius estimate
#         rho_new = xTBx / torch.clamp(xTx, min=1e-7)

#         # Compute relative difference
#         reldiff = torch.abs((rho_new - rho) / torch.clamp(rho, min=1e-7))

#         # Print iteration info
#         # print(f"Iteration {i}: {torch.sum(reldiff > tol)} matrices not converged")

#         # Only update rho where not converged
#         not_converged = reldiff > tol
#         # rho[not_converged] = rho_new[not_converged]

#         #Update eigenvector estimate
#         BxTBx = opt_einsum.contract('Ni,Ni->N', Bx, Bx)
#         x = Bx/torch.clamp(BxTBx[:, None], min=1e-7)

#         i += 1
    
#     print(f"Total iterations: {i}")
    
#     if i == max_iterations:
#         print('Power iteration did not converge for all matrices')
#         rho[not_converged] = 1

#     # Return normalized eigenvector for the first matrix
#     v = x[0] / (torch.sum(x[0] ** 2)) ** 0.5

#     return rho, v


def rho_max_power_iteration(B, rho_0, x_0, tol, max_iterations, dtype, device):
    """
    Estimate the spectral radius (largest eigenvalue in magnitude) of a batch of matrices 
    using the power iteration method.

    Args:
    B : ndarray of shape (N, M, M)
        Batch of N symmetric square matrices of shape (M, M) whose spectral radii are to be estimated.
    x : ndarray of shape (M,)
        Initial guess for the eigenvector (normalized). Tiled across batch.
    tol : float
        Convergence tolerance (relative change in rho).
    max_iterations : int
        Maximum number of iterations.
    dtype : torch.dtype
        Tensor data type (e.g., torch.float32 or torch.float64).
    device : torch.device
        Device (e.g., torch.device('cuda') or 'cpu').

    Returns:
    rho : torch.Tensor of shape (N,)
        Estimated spectral radii for each matrix.
    """
    # Normalize x and tile across batch
    x = x_0 / torch.clamp(torch.linalg.norm(x_0), min=1e-7)
    x = x.repeat(B.shape[0], 1)  # Shape: (N, M)

    rho = rho_0*torch.ones(B.shape[0], dtype=dtype, device=device)

    for i in range(max_iterations):
        # Matrix-vector multiplication: Bx
        Bx = opt_einsum.contract('Nij,Nj->Ni', B, x)

        # Rayleigh quotient: (x^T B x) / (x^T x)
        xTBx = opt_einsum.contract('Ni,Ni->N', x, Bx)
        xTx = opt_einsum.contract('Ni,Ni->N', x, x)
        rho_new = xTBx / torch.clamp(xTx, min=1e-7)

        # Relative difference
        reldiff = torch.abs((rho_new - rho) / torch.clamp(rho, min=1e-7))

        # Update rho where reldiff > tol
        # rho = torch.where(reldiff > tol, rho_new, rho)
        not_converged = reldiff > tol
        rho[not_converged] = rho_new[not_converged]

        # Update x
        # Bx_not_converged = Bx[not_converged]
        # Bx_norm = (opt_einsum.contract('Ni,Ni->N', Bx_not_converged, Bx_not_converged))**(1/2)
        # x_new = x.clone()
        # x_new[not_converged] = Bx_not_converged/torch.clamp(Bx_norm[:, None], min=1e-7)
        # x = x_new

        Bx_norm = (opt_einsum.contract('Ni,Ni->N', Bx, Bx))**(1/2)
        # x = Bx / torch.clamp(Bx_norm[:, None], min=1e-7)
        x_new = x.clone()
        x_new[not_converged] = Bx[not_converged] / torch.clamp(Bx_norm[not_converged, None], min=1e-7)
        x = x_new

        # Check for global convergence
        # if torch.all(reldiff <= tol):
        if torch.sum(not_converged) == 0:
            break
    
    print(f"Total iterations: {i+1}")
    if i == max_iterations - 1:
        print('Power iteration did not fully converge within max_iterations. Number of non-converged matrices:', torch.sum(reldiff > tol).item())
        # If not converged, compute exact spectral radius
        rho[not_converged] = torch.amax(torch.abs(torch.linalg.eigvals(B[not_converged])), axis=-1)
        # rho[reldiff > tol] = torch.amax(torch.abs(torch.linalg.eigvals(B[reldiff > tol])), axis=-1)
        print('Exact computation of spectral radius for non-converged matrices.')
        # rho[reldiff > tol] = 1
    # Return normalized eigenvector for the first matrix
    v = x[0] / (torch.sum(x[0] ** 2)) ** 0.5

    return rho, v



def two_norm_power_iteration(B, norm2_0, x_0, tol, max_iterations, dtype, device, model):
    """
    Estimate the 2-norm (largest singular value) of a batch of matrices 
    using power iteration on B^* B implicitly.

    Args:
        B : torch.Tensor of shape (N, M, K)
            Batch of N matrices (not necessarily square).
        norm2_0 : float
            Initial guess for 2-norm.
        x_0 : torch.Tensor of shape (K,)
            Initial guess for right singular vector (normalized). Tiled across batch.
        tol : float
            Convergence tolerance (relative change in norm).
        max_iterations : int
            Maximum number of iterations.
        dtype : torch.dtype
            Tensor data type (e.g., torch.float32 or torch.float64).
        device : torch.device
            Device (e.g., torch.device('cuda') or 'cpu').

    Returns:
        norm2 : torch.Tensor of shape (N,)
            Estimated 2-norms for each matrix.
        v : torch.Tensor of shape (K,)
            Approximate right singular vector for the first matrix.
    """
    N, M, K = B.shape

    norm2 = norm2_0 * torch.ones(N, dtype=dtype, device=device)

    # norm2 = norm2 + x_0.T @ model.C_0 @ model.M_rr22 @ (B - model.C_0) @ x_0

    x = x_0 / torch.clamp(torch.linalg.norm(x_0), min=1e-7)
    x = x.repeat(N, 1)  # Shape: (N, K)

    for i in range(max_iterations):
        # 1) y = B x          shape (N, M)
        y = torch.einsum('nij,nj->ni', B, x)

        # 2) norm of y (for each matrix in batch)
        norm_y = torch.linalg.norm(y, dim=1)

        # 3) z = B^* y         shape (N, K)
        z = torch.einsum('nij,ni->nj', B, y)

        # Normalize z to get next x
        z_norm = torch.linalg.norm(z, dim=1)
        z_norm = torch.clamp(z_norm, min=1e-7)
        x_new = z / z_norm[:, None]

        # Update norm2 estimate
        norm2_new = norm_y

        # Relative difference for convergence check
        reldiff = torch.abs((norm2_new - norm2) / torch.clamp(norm2, min=1e-7))

        # Update norm2 and x only where not converged
        not_converged = reldiff > tol
        # norm2[not_converged] = norm2_new[not_converged]
        # x[not_converged] = x_new[not_converged]
        x = torch.where(not_converged[:, None], x_new, x)
        norm2 = torch.where(not_converged, norm2_new, norm2)

        if torch.sum(not_converged) == 0:
            break

    print(f"Total iterations: {i+1}")

    if i == max_iterations - 1:
        print('Power iteration did not fully converge within max_iterations. Number of non-converged matrices:', torch.sum(reldiff > tol).item())
        # Fall back to exact computation for non-converged matrices
        norm2[not_converged] = torch.linalg.norm(B[not_converged], ord=2, dim=(1, 2))

    # Return the right singular vector for the first matrix
    v = x[0] / torch.linalg.norm(x[0])

    return norm2, v



def sort_matrices(matrices):
    """
    Sort the rows and columns of each matrix in the batch by decreasing L2 norm.
    
    Args:
        matrices (torch.Tensor): A batch of n x n matrices of shape (batch_size, n, n)
    
    Returns:
        torch.Tensor: A batch of sorted n x n matrices of shape (batch_size, n, n)
        torch.Tensor: Row sorted indices of shape (batch_size, n)
        torch.Tensor: Column sorted indices of shape (batch_size, n)
    """
    batch_size, n, _ = matrices.shape
    # Compute the L2 norm for rows and columns
    row_norms = torch.norm(matrices, dim=2, p=np.inf)  # shape: (batch_size, n)
    col_norms = torch.norm(matrices, dim=1, p=np.inf)  # shape: (batch_size, n)
    # Sort indices by decreasing L2 norm
    row_sorted_indices = torch.argsort(row_norms, dim=1, descending=True)  # shape: (batch_size, n)
    col_sorted_indices = torch.argsort(col_norms, dim=1, descending=True)  # shape: (batch_size, n)
    # Create a batch index tensor
    batch_indices = torch.arange(batch_size)#.unsqueeze(1)  # shape: (batch_size, 1)
    n_indices = torch.arange(n)
    # Sort rows and columns using advanced indexing
    sorted_matrices = matrices[batch_indices[:,None,None], row_sorted_indices[:,:,None], col_sorted_indices[:,None,:]]  # shape: (batch_size, n, n)
    return sorted_matrices, row_sorted_indices, col_sorted_indices

def unsort_matrices(sorted_matrices, row_sorted_indices, col_sorted_indices):
    """
    Unsort the matrices back to their original order.
    
    Args:
        sorted_matrices (torch.Tensor): A batch of sorted n x n matrices of shape (batch_size, n, n)
        row_sorted_indices (torch.Tensor): Row sorted indices of shape (batch_size, n)
        col_sorted_indices (torch.Tensor): Column sorted indices of shape (batch_size, n)
    
    Returns:
        torch.Tensor: A batch of unsorted n x n matrices of shape (batch_size, n, n)
    """
    batch_size, n, _ = sorted_matrices.shape
    # Create inverse indices for rows and columns
    row_sorted_indices_inv = torch.argsort(row_sorted_indices, dim=1)  # shape: (batch_size, n)
    col_sorted_indices_inv = torch.argsort(col_sorted_indices, dim=1)  # shape: (batch_size, n)
    # Create a batch index tensor
    batch_indices = torch.arange(batch_size)#.unsqueeze(1)  # shape: (batch_size, 1)
    # # Unsort rows and columns using advanced indexing
    unsorted_matrices = sorted_matrices[batch_indices[:,None,None], col_sorted_indices_inv[:,:,None], row_sorted_indices_inv[:,None,:]]
    return unsorted_matrices

# # Example usage
# batch_size, n = 1, 2
# # matrices = torch.rand(batch_size, n, n)
# matrices = F
# sorter = NormSorter()
# sorted_matrices, row_sorted_indices, col_sorted_indices = sort_matrices(matrices)

# print("Original matrices:")
# print(matrices)
# print("\nSorted matrices:")
# print(sorted_matrices)

# # Unsort the matrices
# unsorted_matrices = unsort_matrices(sorted_matrices, row_sorted_indices, col_sorted_indices)
# print("\nUnsorted matrices (should match the original):")
# print(unsorted_matrices)



class GaussianRBF(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.params = params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hparams = params['hparams']
        # self.reset_nontrainableparameters()
        self.reset_parameters()
        
    def reset_parameters(self):
        self.mus = nn.Parameter(torch.Tensor(self.output_dim, self.input_dim))
        self.log_sigmas = nn.Parameter(torch.Tensor(self.output_dim))
        nn.init.uniform_(self.mus, 0, 1)
        nn.init.constant_(self.log_sigmas, np.log((1/self.hparams['h'])**(1/self.params['simparams']['d'])))
                                     
    def reset_nontrainableparameters(self):
        mu_0, mu_1 = np.mgrid[0:1:int(self.hparams['h']**(1/self.params['simparams']['d']))*1j, 0:1:int(self.hparams['h']**(1/self.params['simparams']['d']))*1j]        
        mus = np.vstack([mu_0.ravel(), mu_1.ravel()]).T
        self.mus = torch.tensor(mus)
        self.log_sigmas = torch.ones(self.output_dim)*np.log((1/self.hparams['h']**(1/self.params['simparams']['d'])))               
                                     
    def forward(self, x):
        if self.hparams.get('symgroupavg',False)==True:
            mus_temp = torch.einsum('ij,gj->gi', self.mapping, self.mus - 1/2) + 1/2
            d_scaled = (x[:,None,:] - mus_temp[None,:,:])/torch.exp(self.log_sigmas[None,:,None])
        else:
            d_scaled = (x[:,None,:] - self.mus[None,:,:])/torch.exp(self.log_sigmas[None,:,None])
        y = torch.exp(-torch.sum(d_scaled**2, axis=-1)/2)
        if self.hparams.get('norm_basis',False)==True:
            y = y/torch.sum(y, axis=-1)[:,None]
        return y
    
    def grad(self, x):
        if self.hparams.get('symgroupavg',False)==True:
            mus_temp = torch.einsum('ij,gj->gi', self.mapping, self.mus - 1/2) + 1/2
            prefactor = -1/(torch.exp(self.log_sigmas[None,:,None]))**2*(x[:,None,:] - mus_temp[None,:,:])
        else:
            prefactor = -1/(torch.exp(self.log_sigmas[None,:,None]))**2*(x[:,None,:] - self.mus[None,:,:])
        return prefactor*self.forward(x)[:,:,None]

        
class GaussianRBF_NOMAD(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.hparams = params['hparams']
        
    def forward(self, mus, log_sigmas, x):
        if self.hparams.get('symgroupavg',False)==True:
            mus_temp = torch.einsum('ij,gj->gi', self.mapping, mus - 1/2) + 1/2
            d_scaled = (x[:,:,None,:] - mus_temp[None,None,:,:])/torch.exp(log_sigmas[None,None,:,None])
        else:
            d_scaled = (x[:,:,None,:] - mus[:,None,:,:])/torch.exp(log_sigmas[:,None,:,None])
        y = torch.exp(-torch.sum(d_scaled**2, axis=-1))
        if self.hparams.get('norm_basis',False)==True:
            y = y/torch.sum(y, axis=-1)[:,:,None]
        return y
    

class ReshapeLayer(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape
    
    def forward(self, x):
        new_shape = (x.shape[0],) + self.output_shape
        y = x.reshape(new_shape)
        return y
    
class InversionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.output_shape = output_shape
    
    def forward(self, x):
        y = torch.linalg.pinv(x)
        return y
    
    
class PConv(nn.Module):
    def __init__(self, hidden_channels, kernel_size, stride, bias):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, bias=bias))
        self.layers.append(nn.BatchNorm2d(num_features=hidden_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=1, kernel_size=kernel_size, stride=stride, bias=bias))
    
    def forward(self, x):
        P = x
        for layer in self.layers:
            P = layer(P)
        P = P/torch.norm(P)
        y = x + torch.matmul(P,x)
        return y

    
class UpsampleModel(nn.Module):
    def __init__(self, size):
        super(UpsampleModel, self).__init__()
        self.size = size
        
    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode='nearest')
        return x
    

def expand_D8(A):
    return [A, 
            #A.rot90(dims=[-2,-1]), 
            A.rot90(dims=[-2,-1]).rot90(dims=[-2,-1]), 
            #A.rot90(dims=[-2,-1]).rot90(dims=[-2,-1]).rot90(dims=[-2,-1]),
            A.flip(dims=[-1]),
            #A.flip(dims=[-1]).rot90(dims=[-2,-1]),
            A.flip(dims=[-1]).rot90(dims=[-2,-1]).rot90(dims=[-2,-1])]
            #A.flip(dims=[-1]).rot90(dims=[-2,-1]).rot90(dims=[-2,-1]).rot90(dims=[-2,-1])]