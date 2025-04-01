# Copyright 2025 Joost Prins

# 3rd Party
import torch
import numpy as np
import opt_einsum
from torch.nn import functional as F


def weightedrelativeL2(w, u_hat, u):
    L22_diff = torch.sum(w[None,:]*(u_hat - u)**2, axis=-1)
    L2_diff = L22_diff**(1/2)
    L22_norm = torch.sum(w[None,:]*(u)**2, axis=-1)
    L2_norm = L22_norm**(1/2)
    return torch.mean(L2_diff/torch.maximum(L2_norm, 1e-7*torch.ones_like(L2_norm)))

def weightedrelativeL2_set(w, u_hat, u):
    L22_diff = np.sum(w[None,:]*(u_hat - u)**2, axis=-1)
    L2_diff = L22_diff**(1/2)
    L22_norm = np.sum(w[None,:]*(u)**2, axis=-1)
    L2_norm = L22_norm**(1/2)
    return L2_diff/np.maximum(L2_norm, 1e-7*np.ones_like(L2_norm))

def matrixnorm(A_hat, A):
    p = 'fro'
    Frobenius = torch.norm(A_hat - A, dim=(-1,-2), p=p)
    return torch.mean(Frobenius)

def relativematrixnorm(A_hat, A):
    p = 'fro'
    Frobenius = torch.norm(A_hat - A, dim=(-1,-2), p=p)
    norm = torch.norm(A, dim=(-1,-2), p=p)
    return torch.mean(Frobenius/torch.maximum(norm, 1e-7*torch.ones_like(norm)))

def relativeL2_coefficients(M, u_m_hat, u_m):
    Du_m = u_m_hat - u_m
    numerator = opt_einsum.contract('Nm,mn,Nn', Du_m, M, Du_m)
    denominator = opt_einsum.contract('Nm,mn,Nn', u_m, M, u_m)
    relativeL2 = (numerator/torch.maximum(denominator, 1e-7*torch.ones_like(denominator)))**(1/2)
    return torch.mean(relativeL2)


