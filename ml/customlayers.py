import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


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