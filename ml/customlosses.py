import torch
from torch.nn import functional as F

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

def frobeniusnorm(A_hat, A):
    Frobenius = torch.norm(A_hat - A, dim=(-1,-2), p='fro')
    return torch.mean(Frobenius)

def relativefrobeniusnorm(A_hat, A):
    Frobenius = torch.norm(A_hat - A, dim=(-1,-2), p='fro')
    norm = torch.norm(A, dim=(-1,-2), p='fro')
    return torch.mean(Frobenius/torch.maximum(norm, 1e-7*torch.ones_like(norm)))
