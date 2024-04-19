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
        if self.hparams.get('modeltype',False)=='DeepONet':
            self.NLBranch = DeepONetBranch(params, input_dim=2*self.hparams['Q']**params['simparams']['d']+2*self.hparams['Q'], output_dim=self.hparams['h'])
        if self.hparams.get('modeltype',False)=='VarMiON':
            self.NLBranch = NLBranch_VarMiON(params)
            self.LBranch_f = LBranchNet(params, input_dim=self.hparams['Q']**params['simparams']['d'], output_dim=self.hparams['h'])
            self.LBranch_eta = LBranchNet(params, input_dim=2*self.hparams['Q'], output_dim=self.hparams['h'])
        if self.hparams.get('modeltype',False)=='NGO':
            self.NLBranch = NLBranch_NGO(params)
        self.Trunk_test = BSplineBasis2D(knots_x=torch.tensor([0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1]), knots_y=torch.tensor([0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1]), polynomial_order=3, **params)
        self.Trunk_trial = BSplineBasis2D(knots_x=torch.tensor([0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1]), knots_y=torch.tensor([0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1]), polynomial_order=3, **params)
        # self.compute_symgroup()
        self.geometry()
        self = self.to(self.hparams['dtype'])

    def compute_K(self, theta, theta_g):
        # print('Assembling K')
        gradTrunk_test = self.Trunk_test.grad(self.xi_Omega)
        gradTrunk_trial = self.Trunk_trial.grad(self.xi_Omega)
        Trunk_test_g = self.Trunk_test.forward(self.xi_Gamma_g)
        gradTrunk_test_g = self.Trunk_test.grad(self.xi_Gamma_g)
        Trunk_trial_g = self.Trunk_trial.forward(self.xi_Gamma_g)
        gradTrunk_trial_g = self.Trunk_trial.grad(self.xi_Gamma_g)
        K = opt_einsum.contract('q,Nq,qmx,qnx->Nmn', self.w_Omega, theta, gradTrunk_test, gradTrunk_trial)
        K += -opt_einsum.contract('q,qm,qx,Nq,qnx->Nmn', self.w_Gamma_g, Trunk_test_g, self.n_Gamma_g, theta_g, gradTrunk_trial_g)
        K += -opt_einsum.contract('q,qn,qx,Nq,qmx->Nmn', self.w_Gamma_g, Trunk_trial_g, self.n_Gamma_g, theta_g, gradTrunk_test_g)
        return K
    
    def compute_d(self, f, etab, etat):
        # print('Assembling d')
        Trunk_test = self.Trunk_test.forward(self.xi_Omega)
        Trunk_test_b = self.Trunk_test.forward(self.xi_Gamma_b)
        Trunk_test_t = self.Trunk_test.forward(self.xi_Gamma_t)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, Trunk_test, f)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_b, Trunk_test_b, etab)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_t, Trunk_test_t, etat)
        return d

    def forward_NGO(self, theta, f, etab, etat, K, d, psi, x):
        K_inv = self.NLBranch.forward(K)
        u_n = torch.einsum('nij,nj->ni', K_inv, d)
        u_hat = torch.einsum('ni,noi->no', u_n, psi)
        return u_hat
    
    def forward_VarMiON(self, theta, f, etab, etat, K, d, psi, x):
        NLBranch = self.NLBranch.forward(theta)
        eta = torch.zeros((etab.shape[0],2*etab.shape[1]), dtype=self.hparams['dtype'], device=self.device)
        eta[:,:etab.shape[1]] = etab
        eta[:,etab.shape[1]:] = etat
        LBranch = self.LBranch_f.forward(f) + self.LBranch_eta.forward(eta)
        u_n = torch.einsum('nij,nj->ni', NLBranch, LBranch)
        u_hat = torch.einsum('ni,noi->no', u_n, psi)
        return u_hat
    
    def forward_DeepONet(self, theta, f, etab, etat, K, d, psi, x):
        eta = torch.zeros((etab.shape[0],2*etab.shape[1]), dtype=self.hparams['dtype'], device=self.device)
        eta[:,:etab.shape[1]] = etab
        eta[:,etab.shape[1]:] = etat
        inputfuncs = torch.cat((theta,f,eta),dim=1)
        u_n = self.NLBranch.forward(inputfuncs)
        u_hat = torch.einsum('ni,noi->no', u_n, psi)
        return u_hat
    
    def forward_FEM(self, theta, theta_g, f, etab, etat, x):
        K = self.compute_K(theta, theta_g)
        d = self.compute_d(f, etab, etat)
        K_inv = torch.linalg.inv(K)
        u_n = torch.einsum('nij,nj->ni', K_inv, d)
        x_shape = x.shape
        psi = self.Trunk_trial.forward(x.reshape((x_shape[0]*x_shape[1], x_shape[2]))).reshape((x_shape[0],x_shape[1],self.hparams['h'])).to(self.device)
        u_hat = torch.einsum('ni,noi->no', u_n, psi)
        return u_hat
    
    def forward(self, theta, f, etab, etat, K, d, psi, x):
        if self.hparams.get('modeltype',False)=='DeepONet':
            u_hat = self.forward_DeepONet(theta, f, etab, etat, K, d, psi, x)
        if self.hparams.get('modeltype',False)=='VarMiON':
            u_hat = self.forward_VarMiON(theta, f, etab, etat, K, d, psi, x)
        if self.hparams.get('modeltype',False)=='NGO':
            u_hat = self.forward_NGO(theta, f, etab, etat, K, d, psi, x)
        return u_hat
    
    def simforward(self, theta, f, etab, etat, x):
        self.bs = 1
        self.geometry()
        theta_in = torch.tensor(theta(np.array(self.xi_Omega)), dtype=self.hparams['dtype']).tile((self.hparams['batch_size'],1))
        theta_g_in = torch.tensor(theta(np.array(self.xi_Gamma_g)), dtype=self.hparams['dtype']).tile((self.hparams['batch_size'],1))
        f_in = torch.tensor(f(np.array(self.xi_Omega)), dtype=self.hparams['dtype']).tile((self.hparams['batch_size'],1))
        etab_in = torch.tensor(etab(np.array(self.xi_Gamma_b)), dtype=self.hparams['dtype']).tile((self.hparams['batch_size'],1))
        etat_in = torch.tensor(etat(np.array(self.xi_Gamma_t)), dtype=self.hparams['dtype']).tile((self.hparams['batch_size'],1))
        x_in = torch.tensor(x, dtype=self.hparams['dtype']).tile(self.hparams['batch_size'],1,1)
        K_in = self.compute_K(theta_in, theta_g_in)
        d_in = self.compute_d(f_in, etab_in, etat_in)
        x_shape = x_in.shape
        psi_in = self.Trunk_trial.forward(x_in.reshape((x_shape[0]*x_shape[1], x_shape[2]))).reshape((x_shape[0],x_shape[1],self.hparams['h'])).to(self.device)
        u = self.forward(theta_in, f_in, etab_in, etat_in, K_in, d_in, psi_in, x_in)
        u = u[0]
        u = torch.detach(u).cpu()
        u = np.array(u)
        return u
    
    def simforward_FEM(self, theta, f, etab, etat, x):
        self.bs = 1
        self.geometry()
        theta_in = torch.tensor(theta(np.array(self.xi_Omega)), dtype=self.hparams['dtype']).tile((self.hparams['batch_size'],1))
        theta_g_in = torch.tensor(theta(np.array(self.xi_Gamma_g)), dtype=self.hparams['dtype']).tile((self.hparams['batch_size'],1))
        f_in = torch.tensor(f(np.array(self.xi_Omega)), dtype=self.hparams['dtype']).tile((self.hparams['batch_size'],1))
        etab_in = torch.tensor(etab(np.array(self.xi_Gamma_b)), dtype=self.hparams['dtype']).tile((self.hparams['batch_size'],1))
        etat_in = torch.tensor(etat(np.array(self.xi_Gamma_t)), dtype=self.hparams['dtype']).tile((self.hparams['batch_size'],1))
        x_in = torch.tensor(x, dtype=self.hparams['dtype']).tile(self.hparams['batch_size'],1,1)
        u = self.forward_FEM(theta_in, theta_g_in, f_in, etab_in, etat_in, x_in)
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

    def geometry(self):
        if self.hparams['quadrature']=='uniform':
            #Interior
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
            self.xi_Gamma_eta = torch.zeros((2*self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
            self.xi_Gamma_eta[:self.hparams['Q']] = self.xi_Gamma_b
            self.xi_Gamma_eta[self.hparams['Q']:] = self.xi_Gamma_t        
            self.xi_Gamma_g = torch.zeros((2*self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
            self.xi_Gamma_g[:self.hparams['Q']] = self.xi_Gamma_l
            self.xi_Gamma_g[self.hparams['Q']:] = self.xi_Gamma_r        
            self.w_Gamma_b = 1/(self.hparams['Q'])*torch.ones((self.hparams['Q']), device=self.device)
            self.w_Gamma_t = 1/(self.hparams['Q'])*torch.ones((self.hparams['Q']), device=self.device)
            self.w_Gamma_eta = 1/(2*self.hparams['Q'])*torch.ones((2*self.hparams['Q']), device=self.device)
            self.w_Gamma_g = 1/(2*self.hparams['Q'])*torch.ones((2*self.hparams['Q']), device=self.device)   
        if self.hparams['quadrature']=='Gauss-Legendre':
            #Interior
            x, w = np.polynomial.legendre.leggauss(int(self.hparams['Q']/5))
            x = np.array(np.meshgrid(x,x,indexing='ij')).reshape(2,-1).T/2 + 0.5
            w = w/2*0.2
            w = (w*w[:,None]).ravel()
            xi_Omega = []
            w_Omega = []
            for i in range(5):
                for j in range(5):
                    newpts = 0.2*x
                    newpts[:,0] = newpts[:,0] + 0.2*i
                    newpts[:,1] = newpts[:,1] + 0.2*j
                    xi_Omega.append(newpts)
                    w_Omega.append(w)
            w_Omega = np.array(w_Omega).flatten()
            self.w_Omega = torch.tensor(w_Omega, dtype=self.hparams['dtype'], device=self.device)
            xi_Omega = np.array(xi_Omega)
            xi_Omega = xi_Omega.reshape(xi_Omega.shape[0]*xi_Omega.shape[1],xi_Omega.shape[2])
            self.xi_Omega = torch.tensor(xi_Omega, dtype=self.hparams['dtype'], device=self.device)
            #Boundaries
            x, w = np.polynomial.legendre.leggauss(int(self.hparams['Q']/5))
            x = x/2 + 0.5
            w = w/2*0.2
            xi_Gamma_i = []
            w_Gamma_i = []
            for i in range(5):
                xi_Gamma_i.append(0.2*x + 0.2*i)
                w_Gamma_i.append(w)
            xi_Gamma_i = np.array(xi_Gamma_i)
            xi_Gamma_i = xi_Gamma_i.flatten()
            xi_Gamma_i = torch.tensor(xi_Gamma_i, dtype=self.hparams['dtype'], device=self.device)
            w_Gamma_i = np.array(w_Gamma_i).flatten()
            w_Gamma_i = torch.tensor(w_Gamma_i, dtype=self.hparams['dtype'], device=self.device)
            self.xi_Gamma_b = torch.zeros((self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
            self.xi_Gamma_b[:,0] = xi_Gamma_i
            self.xi_Gamma_t = torch.ones((self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
            self.xi_Gamma_t[:,0] = xi_Gamma_i
            self.xi_Gamma_l = torch.zeros((self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
            self.xi_Gamma_l[:,1] = xi_Gamma_i
            self.xi_Gamma_r = torch.ones((self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
            self.xi_Gamma_r[:,1] = xi_Gamma_i
            self.xi_Gamma_eta = torch.zeros((2*self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
            self.xi_Gamma_eta[:self.hparams['Q']] = self.xi_Gamma_b
            self.xi_Gamma_eta[self.hparams['Q']:] = self.xi_Gamma_t        
            self.xi_Gamma_g = torch.zeros((2*self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
            self.xi_Gamma_g[:self.hparams['Q']] = self.xi_Gamma_l
            self.xi_Gamma_g[self.hparams['Q']:] = self.xi_Gamma_r 
            self.w_Gamma_b = w_Gamma_i
            self.w_Gamma_t = w_Gamma_i
            self.w_Gamma_l = w_Gamma_i
            self.w_Gamma_r = w_Gamma_i
            self.w_Gamma_eta = torch.zeros((2*len(w_Gamma_i)), device=self.device)
            self.w_Gamma_eta[:len(w_Gamma_i)] = self.w_Gamma_b
            self.w_Gamma_eta[len(w_Gamma_i):] = self.w_Gamma_t
            self.w_Gamma_g = torch.zeros((2*len(w_Gamma_i)), device=self.device)
            self.w_Gamma_g[:len(w_Gamma_i)] = self.w_Gamma_l
            self.w_Gamma_g[len(w_Gamma_i):] = self.w_Gamma_r
        #Outward normal
        n_b = torch.tensor([0,-1], dtype=self.hparams['dtype'], device=self.device)
        self.n_b = torch.tile(n_b,(self.hparams['Q'],1))
        n_t = torch.tensor([0,1], dtype=self.hparams['dtype'], device=self.device)
        self.n_t = torch.tile(n_t,(self.hparams['Q'],1))
        n_l = torch.tensor([-1,0], dtype=self.hparams['dtype'], device=self.device)
        self.n_l = torch.tile(n_l,(self.hparams['Q'],1))
        n_r = torch.tensor([1,0], dtype=self.hparams['dtype'], device=self.device)
        self.n_r = torch.tile(n_r,(self.hparams['Q'],1))
        self.n_Gamma_eta = torch.zeros((2*self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
        self.n_Gamma_eta[:self.hparams['Q']] = self.n_b
        self.n_Gamma_eta[self.hparams['Q']:] = self.n_t
        self.n_Gamma_g = torch.zeros((2*self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
        self.n_Gamma_g[:self.hparams['Q']] = self.n_l
        self.n_Gamma_g[self.hparams['Q']:] = self.n_r
    
    def compute_symgroup(self):
        R = torch.tensor([[0,-1],[1,0]], dtype=self.hparams['dtype'], device=self.device)
        M = torch.tensor([[1,0],[0,-1]], dtype=self.hparams['dtype'], device=self.device)
        I = torch.tensor([[1,0],[0,1]], dtype=self.hparams['dtype'], device=self.device)
        # self.symgroup = [I, R, R@R, R@R@R, M, R@M, R@R@M, R@R@R@M]
        self.symgroup = [I, R@R, M, R@R@M]
        # self.symgroup_inv =[I, torch.linalg.inv(R), torch.linalg.inv(R@R), torch.linalg.inv(R@R@R), torch.linalg.inv(M), torch.linalg.inv(R@M), torch.linalg.inv(R@R@M), torch.linalg.inv(R@R@R@M)]
        self.symgroup_inv = [I, torch.linalg.inv(R@R), torch.linalg.inv(M), torch.linalg.inv(R@R@M)]

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