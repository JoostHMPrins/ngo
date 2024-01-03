import torch
from torch import nn
import numpy as np


class GaussianRBF(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.params = params
        self.hparams = params['hparams']
        self.mus = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.log_sigmas = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.mus, 0, 1)
        nn.init.constant_(self.log_sigmas, np.log((1/self.hparams['latent_dim'])**(1/self.params['simparams']['d'])))
        
    def forward(self, x):
        if self.hparams.get('symgroupavg',False)==True:
            mus_temp = torch.einsum('ij,gj->gi', self.mapping, self.mus - 1/2) + 1/2
            d_scaled = (x[:,:,None,:] - mus_temp[None,None,:,:])/torch.exp(self.log_sigmas[None,None,:,None])
        else:
            d_scaled = (x[:,:,None,:] - self.mus[None,None,:,:])/torch.exp(self.log_sigmas[None,None,:,None])
        y = torch.exp(-torch.sum(d_scaled**2, axis=-1)/2)
        if self.hparams.get('norm_basis',False)==True:
            y = y/torch.sum(y, axis=-1)[:,:,None]
        return y
    
    def grad(self, x):
        if self.hparams.get('symgroupavg',False)==True:
            mus_temp = torch.einsum('ij,gj->gi', self.mapping, self.mus - 1/2) + 1/2
            prefactor = -1/(torch.exp(self.log_sigmas[None,None,:,None]))**2*(x[:,:,None,:] - mus_temp[None,None,:,:])
        else:
            prefactor = -1/(torch.exp(self.log_sigmas[None,None,:,None]))**2*(x[:,:,None,:] - self.mus[None,None,:,:])
        return prefactor*self.forward(x)[:,:,:,None]
    
    def laplacian(self, x):
        d_scaled = (x[:,:,None,:] - self.mus[None,None,:,:])/torch.exp(self.log_sigmas[None,None,:,None])
        prefactor = -2/(torch.exp(self.log_sigmas[None,None,:])**2 + torch.sum(d_scaled**2, axis=-1))/(torch.exp(self.log_sigmas[None,None,:]))**2
        return prefactor*self.forward(x)

        
        
    

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
    

def expand_D8(A):
    return [A, 
            #A.rot90(dims=[-2,-1]), 
            A.rot90(dims=[-2,-1]).rot90(dims=[-2,-1]), 
            #A.rot90(dims=[-2,-1]).rot90(dims=[-2,-1]).rot90(dims=[-2,-1]),
            A.flip(dims=[-1]),
            #A.flip(dims=[-1]).rot90(dims=[-2,-1]),
            A.flip(dims=[-1]).rot90(dims=[-2,-1]).rot90(dims=[-2,-1])]
            #A.flip(dims=[-1]).rot90(dims=[-2,-1]).rot90(dims=[-2,-1]).rot90(dims=[-2,-1])]