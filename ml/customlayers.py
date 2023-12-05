import torch
from torch import nn
import numpy as np


class GaussianRBF(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.hparams = params['hparams']
        self.mus = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.log_sigmas = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.mus, 0, 1)
        nn.init.constant_(self.log_sigmas, np.log(0.5))
        
    def forward(self, x):
        if self.hparams.get('symgroupavg',False)==True:
            mus_temp = torch.einsum('ij,gj->gi', self.mapping, self.mus - 1/2) + 1/2
            d_scaled = (x[:,:,None,:] - mus_temp[None,None,:,:])/torch.exp(self.log_sigmas[None,None,:,None])
        else:
            d_scaled = (x[:,:,None,:] - self.mus[None,None,:,:])/torch.exp(self.log_sigmas[None,None,:,None])
        y = torch.exp(-torch.sum(d_scaled**2, axis=-1))
        if self.hparams.get('norm_basis',False)==True:
            y = y/torch.sum(y, axis=-1)[:,:,None]
        return y
    

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
    
    
class ResizeLayer2D(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.hparams = params['hparams']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim*self.input_dim, self.output_dim*self.output_dim, bias=False))

    def forward(self, x):
        x = x.flatten(-2,-1)
        for layer in self.layers:
            x = layer(x)
        y = x.reshape((self.hparams['batch_size'], self.output_dim, self.output_dim))
        return y
    

class UnsqueezeLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        # self.axis = axis
        
    def forward(self, x):
        y = x.unsqueeze(1)
        return y
    

class SqueezeLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        
    def forward(self, x):
        y = x.squeeze()
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
    
    