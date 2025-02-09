import torch
from torch.nn import functional as F
import numpy as np
import opt_einsum

def L2_scaled(u_hat, u):
    L2 = torch.norm(u_hat - u, p=2, dim=-1)
    norm = torch.norm(u, p=2, dim=-1)
    return torch.mean(L2/norm)

def weightedL2(w, u_hat, u):
    L22 = torch.sum(w[None,:]*(u_hat - u)**2, axis=-1)
    L2 = L22**(1/2)
    return torch.mean(L2)

def weightedL2squared(w, u_hat, u):
    L22 = torch.sum(w[None,:]*(u_hat - u)**2, axis=-1)
    return torch.mean(L22)

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

def weightedrelativeL1_set(w, u_hat, u):
    L2_diff = np.sum(w[None,:]*np.abs(u_hat - u), axis=-1)
    # L2_diff = L22_diff**(1/2)
    L2_norm = np.sum(w[None,:]*np.abs(u), axis=-1)
    # L2_norm = L22_norm**(1/2)
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

def relativevectornorm(A_hat, A):
    p = 'fro'
    Frobenius = torch.norm(A_hat - A, dim=(-1), p=p)
    norm = torch.norm(A, dim=(-1), p=p)
    return torch.mean(Frobenius/torch.maximum(norm, 1e-7*torch.ones_like(norm)))

def vectornorm(A_hat, A):
    p = 'fro'
    Frobenius = torch.norm(A_hat - A, dim=(-1), p=p)
    return torch.mean(Frobenius)

def relativeL2_coefficients(M, u_m_hat, u_m):
    Du_m = u_m_hat - u_m
    numerator = opt_einsum.contract('Nm,mn,Nn', Du_m, M, Du_m)
    denominator = opt_einsum.contract('Nm,mn,Nn', u_m, M, u_m)
    relativeL2 = (numerator/torch.maximum(denominator, 1e-7*torch.ones_like(denominator)))**(1/2)
    return torch.mean(relativeL2)

def relativeMSE_coefficients(M, u_m_hat, u_m):
    Du_m = u_m_hat - u_m
    numerator = opt_einsum.contract('Nm,mn,Nn', Du_m, M, Du_m)
    denominator = opt_einsum.contract('Nm,mn,Nn', u_m, M, u_m)
    relativeL2 = (numerator/torch.maximum(denominator, 1e-7*torch.ones_like(denominator)))**(1/2)
    return torch.mean(relativeL2)


