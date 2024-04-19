import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
import numpy as np
from BSplines import *
from customlayers import *
from customlosses import *
import opt_einsum


# class CNNBranch(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self.hparams = params['hparams']
#         self.rel_kernel_size = 1/3
#         self.kernel_size = int(self.rel_kernel_size*self.hparams['h'])
#         self.padding = int((self.rel_kernel_size*self.hparams['h'] - 1)/2)
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=self.hparams.get('bias_NLBranch',True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=4))
#         self.layers.append(nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=self.hparams.get('bias_NLBranch',True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=4))
#         self.layers.append(nn.Conv2d(in_channels=4, out_channels=4, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=self.hparams.get('bias_NLBranch',True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=self.hparams.get('bias_NLBranch',True)))
#         if self.hparams['NLB_outputactivation']!=None:
#             self.layers.append(self.hparams['NLB_outputactivation'])

#     def forward(self, x):
#         if self.hparams.get('1/theta',False)==True:
#             x = 1/x
#         if self.hparams.get('scale_invariance',False)==True:
#             x_norm = torch.amax(torch.abs(x), dim=(-1,-2))
#             x = x/x_norm[:,None,None]
#         x = x.unsqueeze(1)
#         for layer in self.layers:
#             x = layer(x)
#         y = x.squeeze()
#         if self.hparams.get('Cholesky',False)==True:
#             L = y.tril()
#             D = torch.matmul(L, L.transpose(-1,-2))
#             y = D
#         if self.hparams.get('scale_invariance',False)==True:
#             if self.hparams.get('1/theta',False)==True:
#                 y = y*x_norm[:,None,None]     
#             else:
#                 y = y/x_norm[:,None,None]     
#         return y

# class CNNBranch(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self.hparams = params['hparams']
#         self.rel_kernel_size = 1/4
#         self.kernel_size = int(self.rel_kernel_size*self.hparams['h'])
#         self.channels = 1
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=self.kernel_size, stride=1, padding=0, bias=self.hparams.get('bias_NLBranch',True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=self.channels))
#         self.layers.append(nn.ConvTranspose2d(in_channels=self.channels, out_channels=self.channels, kernel_size=self.kernel_size, stride=1, padding=0, bias=self.hparams.get('bias_NLBranch',True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=self.channels))
#         self.layers.append(nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=self.kernel_size, stride=1, padding=0, bias=self.hparams.get('bias_NLBranch',True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.ConvTranspose2d(in_channels=self.channels, out_channels=1, kernel_size=self.kernel_size, stride=1, padding=0, bias=self.hparams.get('bias_NLBranch',True)))
#         if self.hparams['NLB_outputactivation']!=None:
#             self.layers.append(self.hparams['NLB_outputactivation'])

#     def forward(self, x):
#         if self.hparams.get('1/theta',False)==True:
#             x = 1/x
#         if self.hparams.get('scale_invariance',False)==True:
#             x_norm = torch.amax(torch.abs(x), dim=(-1,-2))
#             x = x/x_norm[:,None,None]
#         x = x.unsqueeze(1)
#         for layer in self.layers:
#             x = layer(x)
#         y = x.squeeze()
#         if self.hparams.get('Cholesky',False)==True:
#             L = y.tril()
#             D = torch.matmul(L, L.transpose(-1,-2))
#             y = D
#         if self.hparams.get('scale_invariance',False)==True:
#             if self.hparams.get('1/theta',False)==True:
#                 y = y*x_norm[:,None,None]     
#             else:
#                 y = y/x_norm[:,None,None]     
#         return y
    
    
# class CNNBranch(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self.hparams = params['hparams']
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.ConvTranspose2d(in_channels=1, out_channels=10, kernel_size=4, stride=1, padding=0, bias=self.hparams.get('bias_NLBranch',True)))
#         for i in range(int(self.hparams['h']/8)-1):
#             self.layers.append(nn.ReLU())
#             self.layers.append(nn.BatchNorm2d(num_features=10))
#             self.layers.append(nn.Conv2d(in_channels=10, out_channels=10, kernel_size=4, stride=1, padding=0, bias=self.hparams.get('bias_NLBranch',True)))
#             self.layers.append(nn.ReLU())
#             self.layers.append(nn.BatchNorm2d(num_features=10))
#             self.layers.append(nn.ConvTranspose2d(in_channels=10, out_channels=10, kernel_size=4, stride=1, padding=0, bias=self.hparams.get('bias_NLBranch',True)))
#             self.layers.append(nn.ReLU())
#             self.layers.append(nn.BatchNorm2d(num_features=10))
#         self.layers.append(nn.Conv2d(in_channels=10, out_channels=1, kernel_size=4, stride=1, padding=0, bias=self.hparams.get('bias_NLBranch',True)))
#         if self.hparams['NLB_outputactivation']!=None:
#             self.layers.append(self.hparams['NLB_outputactivation'])        

#     def forward(self, x):
#         if self.hparams.get('1/theta',False)==True:
#             x = 1/x
#         if self.hparams.get('scale_invariance',False)==True:
#             x_norm = torch.amax(torch.abs(x), dim=(-1,-2))
#             x = x/x_norm[:,None,None]
#         x = x.unsqueeze(1)
#         for layer in self.layers:
#             x = layer(x)
#         y = x.squeeze()
#         if self.hparams.get('Cholesky',False)==True:
#             L = y.tril()
#             D = torch.matmul(L, L.transpose(-1,-2))
#             y = D
#         if self.hparams.get('scale_invariance',False)==True:
#             if self.hparams.get('1/theta',False)==True:
#                 y = y*x_norm[:,None,None]     
#             else:
#                 y = y/x_norm[:,None,None]     
#         return y


# class NLBranchNet(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self.hparams = params['hparams']
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, bias=self.hparams.get('bias_NLBranch',True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=16))
#         self.layers.append(nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, bias=self.hparams.get('bias_NLBranch',True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=32))
#         self.layers.append(nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, bias=self.hparams.get('bias_NLBranch',True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch',True)))
#         if self.hparams['NLB_outputactivation']!=None:
#             self.layers.append(self.hparams['NLB_outputactivation'])

#     def forward(self, x):
#         if self.hparams.get('1/theta',False)==True:
#             x = 1/x
#         if self.hparams.get('scale_invariance',False)==True:
#             x_norm = torch.amax(torch.abs(x), dim=(-1,-2))
#             x = x/x_norm[:,None,None]
#         x = x.unsqueeze(1)
#         for layer in self.layers:
#             x = layer(x)
#         y = x.squeeze()
#         if self.hparams.get('Cholesky',False)==True:
#             L = y.tril()
#             D = torch.matmul(L, L.transpose(-1,-2))
#             y = D
#         if self.hparams.get('scale_invariance',False)==True:
#             if self.hparams.get('1/theta',False)==True:
#                 y = y*x_norm[:,None,None]     
#             else:
#                 y = y/x_norm[:,None,None]     
#         return y


class NLBranch_NGO(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        
        # Adjusted convolutional layers
        self.layers.append(ReshapeLayer((self.hparams['batch_size'],1,self.hparams['h'],self.hparams['h'])))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=16, out_channels=1, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer((self.hparams['batch_size'],100)))
        self.layers.append(nn.Linear(100,100))
        self.layers.append(nn.ReLU())
        self.layers.append(ReshapeLayer((self.hparams['batch_size'],1,10,10)))
        self.layers.append(nn.ConvTranspose2d(1, 16, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(16, 32, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer((self.hparams['batch_size'],64,64)))
        if self.hparams['NLB_outputactivation'] is not None:
            self.layers.append(self.hparams['NLB_outputactivation'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        y = x
        return y


class NLBranch_VarMiON(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        
        # Adjusted convolutional layers
        self.layers.append(ReshapeLayer((self.hparams['batch_size'],1,12,12)))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=4, out_channels=16, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer((self.hparams['batch_size'],32)))
        self.layers.append(nn.Linear(32,32))
        self.layers.append(ReshapeLayer((self.hparams['batch_size'],32,1,1)))
        self.layers.append(nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(16, 8, kernel_size=4, stride=4, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer((self.hparams['batch_size'],64,64)))
        if self.hparams['NLB_outputactivation'] is not None:
            self.layers.append(self.hparams['NLB_outputactivation'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        y = x
        return y
    
    
class LBranchNet(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, output_dim, bias=self.hparams.get('bias_LBranch',True)))

    def forward(self, x):
        if self.hparams.get('scale_invariance',False)==True:
            x_norm = torch.amax(torch.abs(x), dim=(-1,-2))
            x = x/x_norm[:,None,None]
        x = x.flatten(-2,-1)
        for layer in self.layers:
            x = layer(x)
            y = x
        if self.hparams.get('scale_invariance',False)==True:
            y = y*x_norm[:,None]
        return y
    

class DeepONetBranch(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim,output_dim, bias=self.hparams.get('bias_NLBranch',True)))
        if self.hparams['NLB_outputactivation']!=None:
            self.layers.append(self.hparams['NLB_outputactivation'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            y = x
        return y
    
    
class NGO(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hparams.update(params['hparams'])
        self.bs = self.hparams['batch_size']
        #Branch
        if self.hparams.get('DeepONet',False)==True:
            self.NLBranch = DeepONetBranch(params, input_dim=3*self.hparams['Q']**params['simparams']['d'], output_dim=self.hparams['h'])
        if self.hparams.get('VarMiON',False)==True:
            self.NLBranch = NLBranch_VarMiON(params)
            self.LBranch_f = LBranchNet(params, input_dim=self.hparams['Q']**params['simparams']['d'], output_dim=self.hparams['h'])
            self.LBranch_eta = LBranchNet(params, input_dim=self.hparams['Q']**params['simparams']['d'], output_dim=self.hparams['h'])
        if self.hparams.get('NGO',False)==True:
            self.NLBranch = NLBranch_NGO(params)
        # else:
        #     self.NLBranch = CNNBranch(params)
            #Trunk
        # self.Trunk_test = GaussianRBF(params, input_dim=params['simparams']['d'], output_dim=self.hparams['h'])
        self.Trunk_test = BSplineBasis2D(knots_x=torch.tensor([0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1]), knots_y=torch.tensor([0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1]), polynomial_order=3, **params)
        if self.hparams.get('Petrov-Galerkin',False)==True:
            self.Trunk_trial = GaussianRBF(params, input_dim=params['simparams']['d'], output_dim=self.hparams['h'])
        else:
            self.Trunk_trial = self.Trunk_test
        # self.compute_symgroup()
        self.geometry()
        self = self.to(self.hparams['dtype'])
        
#     def compute_K(self, theta):
#         # print('Assembling K')
#         Trunk_test = self.Trunk_test.forward(self.x_Q.reshape((self.bs*self.hparams['Q']*self.hparams['Q'],self.params['simparams']['d']))).reshape((self.bs,self.hparams['Q'],self.hparams['Q'],self.hparams['h']))
#         gradTrunk_test = self.Trunk_test.grad(self.x_Q.reshape((self.bs*self.hparams['Q']*self.hparams['Q'],self.params['simparams']['d']))).reshape((self.bs,self.hparams['Q'],self.hparams['Q'],self.hparams['h'],self.params['simparams']['d']))
#         Trunk_trial = self.Trunk_trial.forward(self.x_Q.reshape((self.bs*self.hparams['Q']*self.hparams['Q'],self.params['simparams']['d']))).reshape((self.bs,self.hparams['Q'],self.hparams['Q'],self.hparams['h']))
#         gradTrunk_trial = self.Trunk_trial.grad(self.x_Q.reshape((self.bs*self.hparams['Q']*self.hparams['Q'],self.params['simparams']['d']))).reshape((self.bs,self.hparams['Q'],self.hparams['Q'],self.hparams['h'],self.params['simparams']['d']))
#         K = 1/torch.sum(self.xi_Omega)*opt_einsum.contract('Nij,ij,Nijmx,Nijnx->Nmn', theta, self.xi_Omega, gradTrunk_test, gradTrunk_trial)
#         K += -1/torch.sum(self.xi_Gamma_g)*opt_einsum.contract('Nijm,ijx,ij,Nij,Nijnx->Nmn', Trunk_test, self.n, self.xi_Gamma_g, theta, gradTrunk_trial)
#         K += -1/torch.sum(self.xi_Gamma_g)*opt_einsum.contract('Nijn,ijx,ij,Nij,Nijmx->Nmn', Trunk_trial, self.n, self.xi_Gamma_g, theta, gradTrunk_test)
#         return K
    
#     def compute_d(self, f, etab, etat):
#         # print('Assembling d')
#         Trunk_test = self.Trunk_test.forward(self.x_Q.reshape((self.bs*self.hparams['Q']*self.hparams['Q'],self.params['simparams']['d']))).reshape((self.bs,self.hparams['Q'],self.hparams['Q'],self.hparams['h']))
#         d = 1/torch.sum(self.xi_Omega)*opt_einsum.contract('Nijm,ij,Nij->Nm', Trunk_test, self.xi_Omega, f)
#         d += 1/torch.sum(self.xi_Gamma_b)*opt_einsum.contract('Nijm,ij,Nij->Nm', Trunk_test, self.xi_Gamma_b, etab)
#         d += 1/torch.sum(self.xi_Gamma_t)*opt_einsum.contract('Nijm,ij,Nij->Nm', Trunk_test, self.xi_Gamma_t, etat)
#         return d

    def compute_K(self, theta):
        # print('Assembling K')
        Trunk_test = self.Trunk_test.forward(self.xi_Omega)
        gradTrunk_test = self.Trunk_test.grad(self.xi_Omega)
        Trunk_trial = self.Trunk_trial.forward(self.xi_Omega)
        gradTrunk_trial = self.Trunk_trial.grad(self.xi_Omega)
        K = 1/torch.sum(self.xi_Omega)*opt_einsum.contract('Nij,ij,Nijmx,Nijnx->Nmn', theta, self.xi_Omega, gradTrunk_test, gradTrunk_trial)
        K += -1/torch.sum(self.xi_Gamma_g)*opt_einsum.contract('Nijm,ijx,ij,Nij,Nijnx->Nmn', Trunk_test, self.n, self.xi_Gamma_g, theta, gradTrunk_trial)
        K += -1/torch.sum(self.xi_Gamma_g)*opt_einsum.contract('Nijn,ijx,ij,Nij,Nijmx->Nmn', Trunk_trial, self.n, self.xi_Gamma_g, theta, gradTrunk_test)
        return K
    
    def compute_d(self, f, etab, etat):
        # print('Assembling d')
        Trunk_test = self.Trunk_test.forward(self.x_Q.reshape((self.bs*self.hparams['Q']*self.hparams['Q'],self.params['simparams']['d']))).reshape((self.bs,self.hparams['Q'],self.hparams['Q'],self.hparams['h']))
        d = 1/torch.sum(self.xi_Omega)*opt_einsum.contract('Nijm,ij,Nij->Nm', Trunk_test, self.xi_Omega, f)
        d += 1/torch.sum(self.xi_Gamma_b)*opt_einsum.contract('Nijm,ij,Nij->Nm', Trunk_test, self.xi_Gamma_b, etab)
        d += 1/torch.sum(self.xi_Gamma_t)*opt_einsum.contract('Nijm,ij,Nij->Nm', Trunk_test, self.xi_Gamma_t, etat)
        return d

    def forward_NGO(self, theta, f, etab, etat, K, d, psi, x):
        K_inv = self.NLBranch.forward(K)
        u_n = torch.einsum('nij,nj->ni', K_inv, d)
        u_hat = torch.einsum('ni,noi->no', u_n, psi)
        return u_hat
    
    def forward_VarMiON(self, theta, f, etab, etat, K, d, psi, x):
        NLBranch = self.NLBranch.forward(theta)
        LBranch = self.LBranch_f.forward(f) + self.LBranch_eta.forward(etab*self.xi_Gamma_b + etat*self.xi_Gamma_t)
        u_n = torch.einsum('nij,nj->ni', NLBranch, LBranch)
        u_hat = torch.einsum('ni,noi->no', u_n, psi)
        return u_hat
    
    def forward_DeepONet(self, theta, f, etab, etat, K, d, psi, x):
        eta = etab*self.xi_Gamma_b + etat*self.xi_Gamma_t
        theta = theta.flatten(-2,-1)
        f = f.flatten(-2,-1)
        eta = eta.flatten(-2,-1)
        inputfuncs = torch.cat((theta,f,eta),dim=1)
        u_n = self.NLBranch.forward(inputfuncs)
        u_hat = torch.einsum('ni,noi->no', u_n, psi)
        return u_hat
    
    def forward_FEM(self, theta, f, etab, etat, x):
        K = self.compute_K(theta)
        d = self.compute_d(f, etab, etat)
        K_inv = torch.linalg.inv(K)
        u_n = torch.einsum('nij,nj->ni', K_inv, d)
        x_shape = x.shape
        psi = self.Trunk_trial.forward(x.reshape((x_shape[0]*x_shape[1], x_shape[2]))).reshape((x_shape[0],x_shape[1],self.hparams['h'])).to(self.device)
        u_hat = torch.einsum('ni,noi->no', u_n, psi)
        return u_hat
    
    def forward(self, theta, f, etab, etat, K, d, psi, x):
        if self.hparams.get('DeepONet',False)==True:
            u_hat = self.forward_DeepONet(theta, f, etab, etat, K, d, psi, x)
        if self.hparams.get('VarMiON',False)==True:
            u_hat = self.forward_VarMiON(theta, f, etab, etat, K, d, psi, x)
        if self.hparams.get('NGO',False)==True:
            u_hat = self.forward_NGO(theta, f, etab, etat, K, d, psi, x)
        # else:
        #     u_hat = self.forward_NGO(theta, f, etab, etat, K, d, x)
        return u_hat
    
    def simforward(self, theta, f, etab, etat, x):
        self.bs = 1
        self.geometry()
        x_0_Q, x_1_Q = np.mgrid[0:1:self.hparams['Q']*1j, 0:1:self.hparams['Q']*1j]
        x_Q = np.vstack([x_0_Q.ravel(), x_1_Q.ravel()]).T
        theta = torch.tensor(theta(x_Q), dtype=self.hparams['dtype']).reshape((self.hparams['Q'],self.hparams['Q'])).tile((self.hparams['batch_size'],1,1))
        f = torch.tensor(f(x_Q), dtype=self.hparams['dtype']).reshape((self.hparams['Q'],self.hparams['Q'])).tile((self.hparams['batch_size'],1,1))
        etab = torch.tensor(etab(x_Q), dtype=self.hparams['dtype']).reshape((self.hparams['Q'],self.hparams['Q'])).tile((self.hparams['batch_size'],1,1))
        etat = torch.tensor(etat(x_Q), dtype=self.hparams['dtype']).reshape((self.hparams['Q'],self.hparams['Q'])).tile((self.hparams['batch_size'],1,1))
        x = torch.tensor(x, dtype=self.hparams['dtype']).tile(self.hparams['batch_size'],1,1)
        K = self.compute_K(theta)
        d = self.compute_d(f, etab, etat)
        x_shape = x.shape
        psi = self.Trunk_trial.forward(x.reshape((x_shape[0]*x_shape[1], x_shape[2]))).reshape((x_shape[0],x_shape[1],self.hparams['h'])).to(self.device)
        u = self.forward(theta, f, etab, etat, K, d, psi, x)
        u = u[0]
        u = torch.detach(u).cpu()
        u = np.array(u)
        return u
    
    def simforward_FEM(self, theta, f, etab, etat, x):
        self.bs = 1
        self.geometry()
        x_0_Q, x_1_Q = np.mgrid[0:1:self.hparams['Q']*1j, 0:1:self.hparams['Q']*1j]
        x_Q = np.vstack([x_0_Q.ravel(), x_1_Q.ravel()]).T
        theta = torch.tensor(theta(x_Q), dtype=self.hparams['dtype']).reshape((self.hparams['Q'],self.hparams['Q'])).tile((2,1,1))
        f = torch.tensor(f(x_Q), dtype=self.hparams['dtype']).reshape((self.hparams['Q'],self.hparams['Q'])).tile((2,1,1))
        etab = torch.tensor(etab(x_Q), dtype=self.hparams['dtype']).reshape((self.hparams['Q'],self.hparams['Q'])).tile((2,1,1))
        etat = torch.tensor(etat(x_Q), dtype=self.hparams['dtype']).reshape((self.hparams['Q'],self.hparams['Q'])).tile((2,1,1))
        x = torch.tensor(x, dtype=self.hparams['dtype']).tile(2,1,1)
        u = self.forward_FEM(theta, f, etab, etat, x)
        u = u[0]
        u = torch.detach(u).cpu()
        u = np.array(u)
        return u

    def configure_optimizers(self):
        optimizer = self.hparams['optimizer'](self.parameters(), lr=self.hparams['learning_rate'])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        theta, f, etab, etat, K, d, psi, x, u = train_batch
        if self.hparams.get('symgroupavg',False)==True:    
            u_hat = self.symgroupavg_forward(theta, f, etab, etat, K, d, psi, x)
        else:
            u_hat = self.forward(theta, f, etab, etat, K, d, psi, x)
        loss = 0
        for i in range(len(self.hparams['loss_coeffs'])):
            loss = loss + self.hparams['loss_coeffs'][i]*self.hparams['loss_terms'][i](u_hat, u)
        loss = loss/sum(self.hparams['loss_coeffs'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        theta, f, etab, etat, K, d, psi, x, u = val_batch
        if self.hparams.get('symgroupavg',False)==True:    
            u_hat = self.symgroupavg_forward(theta, f, etab, etat, K, d, psi, x)
        else:
            u_hat = self.forward(theta, f, etab, etat, K, d, psi, x)
        loss = 0
        for i in range(len(self.hparams['loss_coeffs'])):
            loss = loss + self.hparams['loss_coeffs'][i]*self.hparams['loss_terms'][i](u_hat, u)
        loss = loss/sum(self.hparams['loss_coeffs']) 
        self.log('val_loss', loss)
        metric = self.hparams['metric'](u_hat, u)
        self.log('metric', metric)
        
    # def geometry(self):
    #     #Domain
    #     self.xi_Omega = torch.ones((self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.device)
    #     x_0_Q, x_1_Q = np.mgrid[0:1:self.hparams['Q']*1j, 0:1:self.hparams['Q']*1j]
    #     x_Q = np.vstack([x_0_Q.ravel(), x_1_Q.ravel()]).T
    #     x_Q = np.tile(x_Q,(self.bs,1,1))
    #     self.x_Q = torch.tensor(x_Q, dtype=self.hparams['dtype'], device=self.device)
    #     #Boundaries
    #     self.xi_Gamma_b = torch.zeros((self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.device)
    #     self.xi_Gamma_b[1:-1,0] = 1
    #     self.xi_Gamma_t = torch.zeros((self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.device)
    #     self.xi_Gamma_t[1:-1,-1] = 1
    #     self.xi_Gamma_l = torch.zeros((self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.device)
    #     self.xi_Gamma_l[0,1:-1] = 1
    #     self.xi_Gamma_r = torch.zeros((self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.device)
    #     self.xi_Gamma_r[-1,1:-1] = 1
    #     self.xi_Gamma = self.xi_Gamma_b + self.xi_Gamma_t + self.xi_Gamma_l + self.xi_Gamma_r
    #     self.xi_Gamma_eta = self.xi_Gamma_b + self.xi_Gamma_t
    #     self.xi_Gamma_g = self.xi_Gamma_l + self.xi_Gamma_r
    #     #Outward normal
    #     self.n = torch.zeros((self.hparams['Q'],self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
    #     self.n[0,1:-1,:] = torch.tensor([-1,0], dtype=self.hparams['dtype'], device=self.device)
    #     self.n[-1,1:-1,:] = torch.tensor([1,0], dtype=self.hparams['dtype'], device=self.device)
    #     self.n[1:-1,0,:] = torch.tensor([0,-1], dtype=self.hparams['dtype'], device=self.device)
    #     self.n[1:-1,-1,:] = torch.tensor([0,1], dtype=self.hparams['dtype'], device=self.device)
    

    def geometry(self):
        #Domain
        x_0_Q, x_1_Q = np.mgrid[0:1:self.hparams['Q']*1j, 0:1:self.hparams['Q']*1j]
        x_Q = np.vstack([x_0_Q.ravel(), x_1_Q.ravel()]).T
        self.xi_Omega = torch.tensor(x_Q, dtype=self.hparams['dtype'], device=self.device)
        self.w_Omega = 1/(self.hparams['Q']**self.params['simparams']['d'])*torch.ones((self.hparams['Q']**self.params['simparams']['d']), device=self.device)
        #Boundaries
        xi_Gamma = torch.linspace(0, 1, self.hparams['Q'], device=self.device)
        self.xi_Gamma_b = torch.zeros((self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
        self.xi_Gamma_b[:,0] = xi_Gamma
        self.xi_Gamma_t = torch.ones((self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
        self.xi_Gamma_t[:,0] = xi_Gamma
        self.xi_Gamma_l = torch.zeros((self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
        self.xi_Gamma_l[:,1] = xi_Gamma
        self.xi_Gamma_r = torch.ones((self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
        self.xi_Gamma_r[:,1] = xi_Gamma
        self.xi_Gamma_eta = torch.tensor([self.xi_Gamma_b, self.xi_Gamma_t]).reshape((2*self.hparams['Q'],self.params['simparams']['d']))
        self.xi_Gamma_g = torch.tensor([self.xi_Gamma_l, self.xi_Gamma_r]).reshape((2*self.hparams['Q'],self.params['simparams']['d']))
        self.w_Gamma_eta = 1/(2*self.hparams['Q'])*torch.ones((2*self.hparams['Q']), device=self.device)
        self.w_Gamma_g = 1/(2*self.hparams['Q'])*torch.ones((2*self.hparams['Q']), device=self.device)    
        #Outward normal
        n_b = torch.tensor([0,-1], dtype=self.hparams['dtype'], device=self.device)
        self.n_b = torch.tile(n_b,(self.hparams['Q'],1))
        n_t = torch.tensor([0,1], dtype=self.hparams['dtype'], device=self.device)
        self.n_t = torch.tile(n_t,(self.hparams['Q'],1))
        n_l = torch.tensor([-1,0], dtype=self.hparams['dtype'], device=self.device)
        self.n_l = torch.tile(n_l,(self.hparams['Q'],1))
        n_r = torch.tensor([1,0], dtype=self.hparams['dtype'], device=self.device)
        self.n_r = torch.tile(n_r,(self.hparams['Q'],1))
        self.n_eta = torch.tensor([self.n_b,self.n_t]).reshape((2*self.hparams['Q'],self.params['simparams']['d']))
        self.n_g = torch.tensor([self.n_l,self.n_r]).reshape((2*self.hparams['Q'],self.params['simparams']['d']))
        
    def geometry_NGO(self):
        #Compute Gauss-Legendre quadrature points and weights
        x, w = np.polynomial.legendre.leggauss(20)
        gauss_pts = np.array(np.meshgrid(x,x,indexing='ij')).reshape(2,-1).T/2 + 0.5
        weights = (w*w[:,None]).ravel()
        x_Q = []
        w = []
        for i in range(5):
            for j in range(5):
                newpts = 0.2*gauss_pts
                newpts[:,0] = newpts[:,0] + 0.2*i
                newpts[:,1] = newpts[:,1] + 0.2*j
                x_Q.append(newpts)
                w.append(weights)
        w = np.array(w).flatten()
        x_Q = np.array(x_Q)
        x_Q = pts.reshape(x_Q.shape[0]*x_Q.shape[1],x_Q.shape[2])
        self.x_Q = torch.tensor(x_Q, dtype=self.hparams['dtype'], device=self.device)
        self.w = torch.tensor(w, dtype=self.hparams['dtype'], device=self.device)
        self.Gamma_eta = torch.zeros(self.x_Q.shape[0], dtype=self.hparams['dtype'], device=self.device)
        self.Gamma_eta[self.x_Q[:,1]==torch.amin(self.x_Q)] = 1
        self.Gamma_eta[self.x_Q[:,1]==torch.amax(self.x_Q)] = 1
        self.Gamma_g = torch.zeros(x_Q.shape[0], dtype=self.hparams['dtype'], device=self.device)
        self.Gamma_g[self.x_Q[:,0]==torch.amin(self.x_Q)] = 1
        self.Gamma_g[self.x_Q[:,0]==torch.amax(self.x_Q)] = 1
        self.n = torch.zeros((self.x_Q.shape[0],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
        self.n[self.x_Q[:,0]==torch.amin(self.x_Q)] = torch.tensor([-1.0,0.0], dtype=self.hparams['dtype'], device=self.device)
        self.n[self.x_Q[:,0]==torch.amax(self.x_Q)] = torch.tensor([1.0,0.0], dtype=self.hparams['dtype'], device=self.device)
        self.n[self.x_Q[:,1]==torch.amin(self.x_Q)] = torch.tensor([0.0,-1.0], dtype=self.hparams['dtype'], device=self.device)
        self.n[self.x_Q[:,1]==torch.amax(self.x_Q)] = torch.tensor([0.0,1.0], dtype=self.hparams['dtype'], device=self.device)
    
    def compute_symgroup(self):
        R = torch.tensor([[0,-1],[1,0]], dtype=self.hparams['dtype'], device=self.device)
        M = torch.tensor([[1,0],[0,-1]], dtype=self.hparams['dtype'], device=self.device)
        I = torch.tensor([[1,0],[0,1]], dtype=self.hparams['dtype'], device=self.device)
        # self.symgroup = [I, R, R@R, R@R@R, M, R@M, R@R@M, R@R@R@M]
        self.symgroup = [I, R@R, M, R@R@M]
        # self.symgroup_inv =[I, torch.linalg.inv(R), torch.linalg.inv(R@R), torch.linalg.inv(R@R@R), torch.linalg.inv(M), torch.linalg.inv(R@M), torch.linalg.inv(R@R@M), torch.linalg.inv(R@R@R@M)]
        self.symgroup_inv = [I, torch.linalg.inv(R@R), torch.linalg.inv(M), torch.linalg.inv(R@R@M)]
    
    def on_fit_start(self):
        self.xi_Omega = self.xi_Omega.to(self.device)
        self.x_Q = self.x_Q.to(self.device)
        self.xi_Gamma_b = self.xi_Gamma_b.to(self.device)
        self.xi_Gamma_t = self.xi_Gamma_t.to(self.device)
        self.xi_Gamma_l = self.xi_Gamma_l.to(self.device)
        self.xi_Gamma_r = self.xi_Gamma_r.to(self.device)
        self.xi_Gamma = self.xi_Gamma.to(self.device)
        self.xi_Gamma_eta = self.xi_Gamma_eta.to(self.device)
        self.xi_Gamma_g = self.xi_Gamma_g.to(self.device)
        self.n = self.n.to(self.device)
        # self.Trunk_test.mus = self.Trunk_test.mus.to(self.device)
        # self.Trunk_test.log_sigmas = self.Trunk_test.log_sigmas.to(self.device)
        # self.Trunk_trial.mus = self.Trunk_trial.mus.to(self.device)
        # self.Trunk_trial.log_sigmas = self.Trunk_trial.log_sigmas.to(self.device)

    def on_before_zero_grad(self, optimizer):
        if self.hparams.get('bound_mus',False)==True:
            for name, p in self.Trunk_test.named_parameters():
                if name=='mus':
                    p.data.clamp_(0, 1.0)
            for name, p in self.Trunk_trial.named_parameters():
                if name=='mus':
                    p.data.clamp_(0, 1.0)
            
    def on_save_checkpoint(self, checkpoint):
        checkpoint['params'] = self.params