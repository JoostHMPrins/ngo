# Copyright (c) 2023-2025 Eindhoven University of Technology

# 3rd Party
import torch
from torch.nn import functional as F
import opt_einsum

# Local
from ngo.numericalanalysis.linalg import integrate, kron_innerproduct


def relativeL2norm_set(W, u_q_hat, u_q):
    """
    Compute the weighted relative L2 norm of the difference between predicted and true values for each sample in the batch.

    Args:
        W (torch.Tensor): Tuple of weights for each dimension. Shape: (Q_i,).
        u_q_hat (torch.Tensor): Predicted values. Shape: (B, Q_0, ..., Q_{d-1}).
        u_q (torch.Tensor): True values. Shape: (B, Q_0, ..., Q_{d-1}).

    Returns:
        torch.Tensor: Weighted relative L2 norms. Shape: (B)
    """
    L2_diff = integrate(W, (u_q_hat - u_q)**2)**(1/2)
    L2_norm = integrate(W, u_q**2)**(1/2)
    return L2_diff/torch.clamp(L2_norm, min=1e-7)


def relativeL2norm(W, u_q_hat, u_q):
    """
    Compute the batch-averaged weighted relative L2 norm of the difference between predicted and true values.

    Args:
        W (torch.Tensor): Tuple of weights for each dimension. Shape: (Q_i,).
        u_q_hat (torch.Tensor): Predicted values. Shape: (B, Q_0, ..., Q_{d-1}).
        u_q (torch.Tensor): True values. Shape: (B, Q_0, ..., Q_{d-1}).
    
    Returns:
        torch.Tensor: Weighted relative L2 norm. Shape: ().
    """
    # Compute weighted relative L2 loss
    return torch.mean(relativeL2norm_set(W, u_q_hat, u_q))


def relativeL2norm_coefficients_set(M_i, u_m_hat, u_m):
    """
    Compute the weighted relative L2 norm in coefficient space for each sample in the batch.

    Args:
        M_i (list of torch.Tensor): List of mass matrices for each dimension. Each of shape (N_i, N_i).
        u_m_hat (torch.Tensor): Predicted coefficients. Shape: (B, N).
        u_m (torch.Tensor): True coefficients. Shape: (B, N).

    Returns:
        torch.Tensor: Weighted relative L2 norms. Shape: (B).
    """
    Du = u_m_hat - u_m
    L2_diff = kron_innerproduct(Du, M_i, Du)**(1/2)
    L2_norm = kron_innerproduct(u_m, M_i, u_m)**(1/2)
    relativeL2norm = L2_diff / torch.clamp(L2_norm, min=1e-7)
    return relativeL2norm


def relativeL2norm_coefficients(M_i, u_m_hat, u_m):
    """
    Compute the batch-averaged weighted relative L2 norm in coefficient space.

    Args:
        M_i (list of torch.Tensor): List of mass matrices for each dimension. Each of shape (N_i, N_i).
        u_m_hat (torch.Tensor): Predicted coefficients. Shape: (B, N).
        u_m (torch.Tensor): True coefficients. Shape: (B, N).

    Returns:
        torch.Tensor: Weighted relative L2 norm. Shape: ().
    """
    # Compute weighted relative L2 loss
    return torch.mean(relativeL2norm_coefficients_set(M_i, u_m_hat, u_m))


def matrixnorm(A_hat, A):
    """
    Compute the Frobenius norm of the difference between two matrices.

    Args:
        A_hat (torch.Tensor): Predicted matrix. Shape: (batch_size, n_rows, n_cols).
        A (torch.Tensor): True matrix. Shape: (batch_size, n_rows, n_cols).

    Returns:
        torch.Tensor: Frobenius norm of the difference. Shape: ().
    """
    p = 'fro'
    Frobenius = torch.norm(A_hat - A, dim=(-1,-2), p=p)
    return torch.mean(Frobenius)


def relativematrixnorm(A_hat, A):
    """
    Compute the relative Frobenius norm of the difference between two matrices.

    Args:
        A_hat (torch.Tensor): Predicted matrix. Shape: (batch_size, n_rows, n_cols).
        A (torch.Tensor): True matrix. Shape: (batch_size, n_rows, n_cols).

    Returns:
        torch.Tensor: Relative Frobenius norm of the difference. Shape: ().
    """
    p = 'fro'
    Frobenius = torch.norm(A_hat - A, dim=(-1,-2), p=p)
    norm = torch.norm(A, dim=(-1,-2), p=p)
    return torch.mean(Frobenius/torch.maximum(norm, 1e-7*torch.ones_like(norm)))


