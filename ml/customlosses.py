import torch
from torch.nn import functional as F

def L2_scaled(u_hat, u):
    L2 = torch.norm(u_hat - u, p=2, dim=-1)
    norm = torch.norm(u, p=2, dim=-1)
    return torch.mean(L2/norm)

    
