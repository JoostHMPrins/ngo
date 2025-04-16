# Copyright 2025 Joost Prins

# 3rd Party
import torch
import numpy as np
import opt_einsum
from torch.nn import functional as F


def weightedrelativeL2(w, u_hat, u):
    """
    Compute the weighted relative L2 loss between predicted and true values. Used as training loss.

    Args:
        w (torch.Tensor): Weights for each dimension. Shape: (n_features,).
        u_hat (torch.Tensor): Predicted values. Shape: (batch_size, n_features).
        u (torch.Tensor): True values. Shape: (batch_size, n_features).

    Returns:
        torch.Tensor: Weighted relative L2 loss. Shape: ().
    """
    L22_diff = torch.sum(w[None,:]*(u_hat - u)**2, axis=-1)
    L2_diff = L22_diff**(1/2)
    L22_norm = torch.sum(w[None,:]*(u)**2, axis=-1)
    L2_norm = L22_norm**(1/2)
    return torch.mean(L2_diff/torch.maximum(L2_norm, 1e-7*torch.ones_like(L2_norm)))

def weightedrelativeL2_set(w, u_hat, u):
    """
    Compute the weighted relative L2 loss for a set of predictions and true values. Used upon testing the model.

    Args:
        w (np.ndarray): Weights for each dimension. Shape: (n_features,).
        u_hat (np.ndarray): Predicted values. Shape: (batch_size, n_features).
        u (np.ndarray): True values. Shape: (batch_size, n_features).

    Returns:
        np.ndarray: Weighted relative L2 loss for each sample. Shape: (batch_size,).
    """
    L22_diff = np.sum(w[None,:]*(u_hat - u)**2, axis=-1)
    L2_diff = L22_diff**(1/2)
    L22_norm = np.sum(w[None,:]*(u)**2, axis=-1)
    L2_norm = L22_norm**(1/2)
    return L2_diff/np.maximum(L2_norm, 1e-7*np.ones_like(L2_norm))

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

def relativeL2_coefficients(M, u_m_hat, u_m):
    """
    Compute the relative L2 loss when a model is trained on projection coefficients instead of on solution values.

    Args:
        M (torch.Tensor): Mass matrix corresponding to the used basis. Shape: (n_features, n_features).
        u_m_hat (torch.Tensor): Predicted coefficients. Shape: (batch_size, n_features).
        u_m (torch.Tensor): True coefficients. Shape: (batch_size, n_features).
        
    Returns:
        torch.Tensor: Relative L2 loss for coefficients. Shape: ().
    """
    Du_m = u_m_hat - u_m
    numerator = opt_einsum.contract('Nm,mn,Nn', Du_m, M, Du_m)
    denominator = opt_einsum.contract('Nm,mn,Nn', u_m, M, u_m)
    relativeL2 = (numerator/torch.maximum(denominator, 1e-7*torch.ones_like(denominator)))**(1/2)
    return torch.mean(relativeL2)


