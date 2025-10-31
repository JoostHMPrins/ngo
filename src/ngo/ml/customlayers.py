# Copyricht 2025 Joost Prins

# 3rd Party
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from joblib import Parallel, delayed
import opt_einsum


class ReshapeLayer(nn.Module):
    """
    A custom PyTorch layer to reshape the input tensor to a specified output shape.

    Attributes:
        output_shape (tuple): The desired shape of the output tensor (excluding the batch dimension).
    """
    def __init__(self, output_shape):
        """
        Initialize the ReshapeLayer.

        Args:
            output_shape (tuple): The desired shape of the output tensor (excluding the batch dimension).
        """
        super().__init__()
        self.output_shape = output_shape
    
    def forward(self, x):
        """
        Reshape the input tensor to the specified output shape.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, ...).

        Returns:
            torch.Tensor: Reshaped tensor. Shape: (batch_size, *output_shape).
        """
        new_shape = (x.shape[0],) + self.output_shape
        y = x.reshape(new_shape)
        return y
    

def discretize_functions(f_list, x):
    """
    Discretize a list of functions over a given input.

    Args:
        f_list (list): A list of callable functions.
        x (np.ndarray): Input points to evaluate the functions. Shape: (n_points,).

    Returns:
        np.ndarray: Discretized function values. Shape: (len(f_list), n_points).
    """
    f_discretized = np.zeros((len(f_list),x.shape[0]))
    for i in range(len(f_list)):
        f_discretized[i] = f_list[i](x)
    # f_discretized = Parallel(n_jobs=-1)(delayed(f)(x) for f in f_list)
    # f_discretized = np.array(f_discretized)
    return f_discretized


def balance_num_trainable_params(model, N_w):
    """
    Adjust the number of trainable parameters in a model to match a target budget.
    This function iteratively adjusts the `free_parameter` attribute of the model
    to ensure the total number of trainable parameters matches the target `N_w`.
    'N_w' serves as an upper limit for the number of trainable parameters: 
    the model will thus have at most 'N_w' trainable parameters.
    The model's `init_layers` method is called to reinitialize the layers after each adjustment.

    Args:
        model (torch.nn.Module): The model to adjust.
        N_w (int): Target number of trainable parameters.

    Returns:
        torch.nn.Module: The adjusted model with the desired number of trainable parameters.
        
    Raises:
        ValueError: If the model cannot meet the trainable parameter budget.
    """
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


def Neumann_preconditioner_forward(model, F):
    """
    Apply Neumann series preconditioning to the input tensor F using the model's preconditioner A_0.
    Args:
        model (torch.nn.Module): The model containing the necessary matrices.
        F (torch.Tensor): Input matrix.
    Returns:
        torch.Tensor: Preconditioned output of the system net A_hat.
    """
    T = torch.zeros(F.shape, dtype=model.dtype, device=model.device)
    Ti = model.Identity
    T1 = -F@model.A_0 + model.Identity
    for i in range(0, model.hparams['Neumannseries_order']):
        Ti = T1@Ti
        T = T + Ti
    A_hat = model.A_0@(model.Identity + T + model.systemnet.forward(T1))
    return A_hat


def conserve_mass(C_m, C, u_m_hat):
    """
    Enforces mass conservation constraint on model predictions.
    This function modifies the model predictions to ensure they satisfy mass conservation.

    Args:
        C_m (torch.Tensor): Mass conservation constraint vector, shape (N, n)
        C (torch.Tensor): Mass conservation constant, shape (N,)
        u_m_hat (torch.Tensor): Model prediction to be constrained, shape (N, n)

    Returns:
        torch.Tensor: Modified prediction u_m_hat satisfying mass conservation, shape (N, n)    
    """
    #Predicted conservation constant
    C_hat = opt_einsum.contract('Nn,Nn->N', u_m_hat, C_m)
    #Scalar correction factor
    l = (C - C_hat)/opt_einsum.contract('Nn,Nn->N', C_m, C_m)
     #Apply mass conservation correction
    u_m_hat = u_m_hat + l[:,None]*C_m
    return u_m_hat


def conserve_energy(model, F, d_x, u_0_m, u_m_hat):
    """
    Enforces energy conservation constraint on model predictions using a quadratic programming approach.
    
    This function modifies the model predictions to ensure they satisfy energy conservation laws
    by solving a quadratic problem. It uses the model's mass matrix to compute
    energy-preserving scaling factors.

    Args:
        model (nn.Module): Neural network model containing mass matrix M_rr
        F (torch.Tensor): NGO input, shape (N, n)
        d_x (torch.Tensor): Spatial part of the right-hand side vector, shape (N, n)
        u_0_m (torch.Tensor): Initial condition, shape (N, n)
        u_m_hat (torch.Tensor): Model prediction to be constrained, shape (N, n)
    
    Returns:
        torch.Tensor: Modified prediction u_m_hat satisfying energy conservation, shape (N, n)
        
    Note:
        N is the batch size, n is the number of Galerkin dofs
    """
    # Get mass matrix from model
    M = model.M_rr
    # Calculate energy conservation constraint terms
    uTd = opt_einsum.contract('Nn,Nn->N', u_m_hat, d_x)
    uTMu = opt_einsum.contract('Nn,mn,Nn->N', u_m_hat, M, u_m_hat)
    uTu = opt_einsum.contract('Nn,Nn->N', u_m_hat, u_m_hat)
    u_im1TMu_im1 = opt_einsum.contract('Nn,mn,Nn->N', u_0_m, M, u_0_m)
    # Identify non-dissipative solutions that violate energy conservation
    non_dissipative = 1/2*uTMu - 1/2*u_im1TMu_im1 > -model.hparams['alpha']/model.hparams['Dt']*uTu + uTd
    # Compute quadratic programming coefficients A, B, C for energy conservation
    A = 1/2*uTMu + model.hparams['alpha']/model.hparams['Dt']*uTu
    B = -uTd
    C = -1/2*u_im1TMu_im1
    # Solve quadratic equation for scaling factors
    a1 = 1/torch.clamp(2*A, min=1e-7)*(-B + torch.sqrt(B**2 - 4*A*C))
    a2 = 1/torch.clamp(2*A, min=1e-7)*(-B - torch.sqrt(B**2 - 4*A*C))
    # Select valid scaling factor (non-negative and minimal)
    a1[a1<0]=torch.inf
    a2[a2<0]=torch.inf
    a_plus = torch.min(a1,a2)
    # Apply conservation constraints only to non-dissipative solutions
    a = torch.ones(u_m_hat.shape[0], dtype=model.dtype, device=model.device)   
    a[non_dissipative] = a_plus[non_dissipative]
    # Compute final correction coefficients and apply to solution
    u_m_hat = a[:,None]*u_m_hat
    return u_m_hat


def conserve_mass_and_energy(model, d_x, u_0_m, C_m, C, u_m_hat):
    """
    Enforces both mass and energy conservation constraints on model predictions using a quadratic programming approach.
    
    This function modifies the model predictions to ensure they satisfy both mass and energy conservation laws
    by solving a quadratic problem. It uses the model's mass matrix and identity operator
    to compute conservation-preserving coefficients.

    Args:
        model (nn.Module): Neural network model containing mass matrix M_rr and identity operator
        d_x (torch.Tensor): Spatial part of the right-hand side vector, shape (N, n)
        u_0_m (torch.Tensor): Initial condition, shape (N, n)
        C_m (torch.Tensor): Mass conservation constraint vector, shape (N, n)
        C (torch.Tensor): Mass conservation constant, shape (N,)
        u_m_hat (torch.Tensor): Model prediction to be constrained, shape (N, n)
    
    Returns:
        torch.Tensor: Modified prediction u_m_hat satisfying conservation laws, shape (N, n)
        
    Note:
        N is the batch size, n is the number of Galerkin dofs
    """
    # Initialize quantities
    d_x = d_x
    C_m = C_m
    C = C
    M = model.M_rr
    # Calculate conservation constraint terms
    CTC = opt_einsum.contract('Nn,Nn->N', C_m, C_m)
    CTd = opt_einsum.contract('Nn,Nn->N', C_m, d_x)
    uTd = opt_einsum.contract('Nn,Nn->N', u_m_hat, d_x)
    C_hat = opt_einsum.contract('Nn,Nn->N', u_m_hat, C_m)
    u_im1TMu_im1 = opt_einsum.contract('Nn,mn,Nn->N', u_0_m, M, u_0_m)
 # Get dissipation parameter and construct augmented mass matrix Q
    alpha =  model.hparams['alpha']
    Q = 1/2*M + alpha*model.Identity/model.hparams['Dt']
    # Calculate quadratic terms for optimization
    uTQu = opt_einsum.contract('Nm,mn,Nn->N', u_m_hat, Q, u_m_hat)
    uTQC = opt_einsum.contract('Nm,mn,Nn->N', u_m_hat, Q, C_m)
    CTQu = opt_einsum.contract('Nm,mn,Nn->N', C_m, Q, u_m_hat)
    CTQC = opt_einsum.contract('Nn,mn,Nn->N', C_m, Q, C_m)
    # Identify non-dissipative solutions that violate energy conservation
    non_dissipative = uTQu > 1/2*u_im1TMu_im1 + uTd
    # Compute quadratic programming coefficients A, B, D for conservation constraints
    A = opt_einsum.contract('Nm,mn,Nn->N', u_m_hat - (C_hat/CTC)[:,None]*C_m, Q, u_m_hat - (C_hat/CTC)[:,None]*C_m)
    B = C/CTC*(uTQC + CTQu) - 2*C*C_hat/CTC**2*CTQC - uTd + C_hat/CTC*CTd
    D = (C/CTC)**2*CTQC - C/CTC*CTd - 1/2*u_im1TMu_im1
    # Solve quadratic equation for scaling factors
    a1 = 1/torch.clamp(2*A, min=1e-7)*(-B + torch.sqrt(B**2 - 4*A*D))
    a2 = 1/torch.clamp(2*A, min=1e-7)*(-B - torch.sqrt(B**2 - 4*A*D))
    # Select valid scaling factor (non-negative and minimal)
    a1[a1<0]=torch.inf
    a2[a2<0]=torch.inf
    a_plus = torch.min(a1,a2)
    # Apply conservation constraints only to non-dissipative solutions
    a = torch.ones(u_m_hat.shape[0], dtype=model.dtype, device=model.device)
    a[non_dissipative] = a_plus[non_dissipative]
    # Compute final correction coefficients and apply to solution
    b = C/CTC - C_hat/CTC*a
    u_m_hat = a[:,None]*u_m_hat + b[:,None]*C_m
    return u_m_hat


def two_norm_power_iteration(B, norm2_0, x_0, tol, max_iterations, dtype, device):
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


def M_norm_scaling(model, A_hat):
    """
    Scale the operator A_hat based on the 2-norm of a transformed version of A_hat.
    Args:
        model (nn.Module): The model containing necessary matrices and hyperparameters.
        A_hat (torch.Tensor): Input operator to be scaled, shape (N, M, K).
    Returns:
        torch.Tensor: Scaled operator A_hat.
    """
    #Compute update matrix
    B = A_hat@model.M_lr
    #Extract bottom-right block
    B = B[:,int(A_hat.shape[1]/2):,int(A_hat.shape[2]/2):]
    #Transform with mass matrix
    M_rr22_05TBM_rr22_05_inv = model.M_rr22_05 @ B @ model.M_rr22_05_inv
    #Estimate 2-norm via power iteration
    Mnorm, v = two_norm_power_iteration(B=M_rr22_05TBM_rr22_05_inv, norm2_0=model.Mnorm_0, x_0=model.x_0, tol=model.hparams['tol'], max_iterations=model.hparams['max_iterations'], dtype=model.dtype, device=model.device)
    #Scale A_hat based on estimated norm
    scaling = torch.maximum(model.hparams['Mnorm_max']*torch.ones_like(Mnorm),Mnorm)/model.hparams['Mnorm_max']
    A_hat = A_hat/scaling[:,None,None]
    return A_hat
    
class InversionLayer(nn.Module):
    """
    Custom PyTorch layer that computes the pseudo-inverse of input tensors.
    
    This layer uses torch.linalg.pinv to compute the Moore-Penrose pseudo-inverse
    of the input tensor.

    Inherits:
        nn.Module: Base PyTorch module class

    Example:
        >>> layer = InversionLayer()
        >>> x = torch.randn(32, 10, 10)  # batch of 32 10x10 matrices
        >>> y = layer(x)  # y will contain the pseudo-inverse of each matrix in x
    """
    def __init__(self):
        """
        Initialize the InversionLayer.
        No parameters are needed for initialization.
        """
        super().__init__()
    
    def forward(self, x):
        """
        Compute the pseudo-inverse of the input tensor.

        Args:
            x (torch.Tensor): Input tensor to be inverted.
                Shape can be (..., m, n) where ... represents any number of batch dimensions.

        Returns:
            torch.Tensor: Pseudo-inverse of the input tensor.
                Shape will be (..., n, m) for input shape (..., m, n).

        Note:
            Uses torch.linalg.pinv which is numerically stable and handles singular matrices.
        """
        y = torch.linalg.pinv(x)
        return y