import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
import numpy as np
from customlayers import *
from customlosses import *
    
class CNNBranch(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.rel_kernel_size = 1/3
        self.kernel_size = int(self.rel_kernel_size*self.hparams['h'])
        self.padding = int((self.rel_kernel_size*self.hparams['h'] - 1)/2)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=self.hparams.get('bias_NLBranch',True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=4))
        self.layers.append(nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=self.hparams.get('bias_NLBranch',True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=4))
        self.layers.append(nn.Conv2d(in_channels=4, out_channels=4, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=self.hparams.get('bias_NLBranch',True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=self.hparams.get('bias_NLBranch',True)))
        if self.hparams['NLB_outputactivation']!=None:
            self.layers.append(self.hparams['NLB_outputactivation'])

    def forward(self, x):
        if self.hparams.get('1/theta',False)==True:
            x = 1/x
        if self.hparams.get('scale_invariance',False)==True:
            x_norm = torch.amax(torch.abs(x), dim=(-1,-2))
            x = x/x_norm[:,None,None]
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        y = x.squeeze()
        if self.hparams.get('Cholesky',False)==True:
            L = y.tril()
            D = torch.matmul(L, L.transpose(-1,-2))
            y = D
        if self.hparams.get('scale_invariance',False)==True:
            if self.hparams.get('1/theta',False)==True:
                y = y*x_norm[:,None,None]     
            else:
                y = y/x_norm[:,None,None]     
        return y
    

class NLBranchNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        self.layers.append(nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch',True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=16))
        self.layers.append(nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch',True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=32))
        self.layers.append(nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch',True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch',True)))
        if self.hparams['NLB_outputactivation']!=None:
            self.layers.append(self.hparams['NLB_outputactivation'])

    def forward(self, x):
        if self.hparams.get('1/theta',False)==True:
            x = 1/x
        if self.hparams.get('scale_invariance',False)==True:
            x_norm = torch.amax(torch.abs(x), dim=(-1,-2))
            x = x/x_norm[:,None,None]
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        y = x.squeeze()
        if self.hparams.get('Cholesky',False)==True:
            L = y.tril()
            D = torch.matmul(L, L.transpose(-1,-2))
            y = D
        if self.hparams.get('scale_invariance',False)==True:
            if self.hparams.get('1/theta',False)==True:
                y = y*x_norm[:,None,None]     
            else:
                y = y/x_norm[:,None,None]     
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
    
    
class NGO(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hparams.update(params['hparams'])
        self.NLBranch = CNNBranch(params)
        if self.hparams.get('VarMiON',False)==True:
            self.LBranch_f = LBranchNet(params, input_dim=self.hparams['Q']**2, output_dim=self.hparams['h'])
            self.LBranch_eta = LBranchNet(params, input_dim=self.hparams['Q']**2, output_dim=self.hparams['h'])
        self.Trunk_test = GaussianRBF(params, input_dim=params['simparams']['d'], output_dim=self.hparams['h'])
        if self.hparams.get('Petrov-Galerkin',False)==True:
            self.Trunk_trial = GaussianRBF(params, input_dim=params['simparams']['d'], output_dim=self.hparams['h'])
        else:
            self.Trunk_trial = self.Trunk_test
        self.compute_symgroup()
        self.geometry()
        self = self.to(self.hparams['dtype'])
        
    def compute_K(self, theta, x):
        Trunk_test = self.Trunk_test.forward(self.x_Q).reshape((self.hparams['batch_size'],self.hparams['Q'],self.hparams['Q'],self.hparams['h']))
        gradTrunk_test = self.Trunk_test.grad(self.x_Q).reshape((self.hparams['batch_size'],self.hparams['Q'],self.hparams['Q'],self.hparams['h'],self.params['simparams']['d']))
        Trunk_trial = self.Trunk_trial.forward(self.x_Q).reshape((self.hparams['batch_size'],self.hparams['Q'],self.hparams['Q'],self.hparams['h']))
        gradTrunk_trial = self.Trunk_trial.grad(self.x_Q).reshape((self.hparams['batch_size'],self.hparams['Q'],self.hparams['Q'],self.hparams['h'],self.params['simparams']['d']))
        # K += -1/torch.sum(self.xi_Omega)*torch.einsum('Nij,ij,Nijmx,Nijnx->Nmn', theta, self.xi_Omega, gradTrunk_test, gradTrunk_trial)
        # K += 1/torch.sum(self.xi_Gamma - self.xi_Gamma_eta)*torch.einsum('Nijm,ijx,ij,Nij,Nijnx->Nmn', Trunk_test, self.n, self.xi_Gamma - self.xi_Gamma_eta, theta, gradTrunk_trial)
        # K += 1/torch.sum(self.xi_Gamma_g)*torch.einsum('Nijn,ijx,ij,Nij,Nijmx->Nmn', Trunk_trial, self.n, self.xi_Gamma_g, theta, gradTrunk_test)
        #Volume term
        # K = -1/torch.sum(self.xi_Omega)*torch.sum(theta[:,:,:,None,None,None]*self.xi_Omega[None,:,:,None,None,None]*gradTrunk_test[:,:,:,:,None,:]*gradTrunk_trial[:,:,:,None,:,:], axis=(1,2,5))
        # #Boundary term 1
        # K += 1/torch.sum(self.xi_Gamma - self.xi_Gamma_eta)*torch.sum(Trunk_test[:,:,:,:,None,None]*self.n[None,:,:,None,None,:]*(self.xi_Gamma - self.xi_Gamma_eta)[None,:,:,None,None,None]*theta[:,:,:,None,None,None]*gradTrunk_trial[:,:,:,None,:,:], axis=(1,2,5))
        # #Boundary term 2
        # K += 1/torch.sum(self.xi_Gamma_g)*torch.sum(Trunk_trial[:,:,:,None,:,None]*self.n[None,:,:,None,None,:]*self.xi_Gamma_g[None,:,:,None,None,None]*theta[:,:,:,None,None,None]*gradTrunk_test[:,:,:,:,None,:], axis=(1,2,5))
        T1 = torch.sum(theta[:,:,:,None,None]*self.xi_Omega[None,:,:,None,None]*gradTrunk_test, axis=(1,2,4))
        T2 = torch.sum(gradTrunk_trial, axis=(1,2,4))
        K = -1/torch.sum(self.xi_Omega)*T1[:,:,None]*T2[:,None,:]
        # K += -1/torch.sum(self.xi_Omega)*torch.einsum('Nm,Nn->Nmn', torch.sum(theta[:,:,:,None,None]*self.xi_Omega[None,:,:,None,None]*gradTrunk_test, axis=(1,2,4)), torch.sum(gradTrunk_trial, axis=(1,2,4)))
        T1 = torch.sum(Trunk_test[:,:,:,:,None]*self.n[None,:,:,None,:]*(self.xi_Gamma - self.xi_Gamma_eta)[None,:,:,None,None]*theta[:,:,:,None,None], axis=(1,2,4))
        T2 = torch.sum(gradTrunk_trial, axis=(1,2,4))
        K += 1/torch.sum(self.xi_Gamma - self.xi_Gamma_eta)*T1[:,:,None]*T2[:,None,:]
        # K += 1/torch.sum(self.xi_Gamma - self.xi_Gamma_eta)*torch.einsum('Nm,Nn->Nmn', torch.sum(Trunk_test[:,:,:,:,None]*self.n[None,:,:,None,:]*(self.xi_Gamma - self.xi_Gamma_eta)[None,:,:,None,None]*theta[:,:,:,None,None], axis=(1,2,4)), torch.sum(gradTrunk_trial, axis=(1,2,4)))
        T1 = torch.sum(Trunk_trial[:,:,:,:,None]*self.n[None,:,:,None,:]*self.xi_Gamma_g[None,:,:,None,None]*theta[:,:,:,None,None], axis=(1,2,4))
        T2 = torch.sum(gradTrunk_test, axis=(1,2,4))
        K += 1/torch.sum(self.xi_Gamma_g)*T1[:,:,None]*T2[:,None,:]
        # K += 1/torch.sum(self.xi_Gamma_g)*torch.einsum('Nn,Nm->Nmn', torch.sum(Trunk_trial[:,:,:,:,None]*self.n[None,:,:,None,:]*self.xi_Gamma_g[None,:,:,None,None]*theta[:,:,:,None,None], axis=(1,2,4)), torch.sum(gradTrunk_test, axis=(1,2,4)))       
        return K
    
    def compute_d(self, f, etab, etat, x):
        Trunk_test = self.Trunk_test.forward(self.x_Q).reshape((self.hparams['batch_size'],self.hparams['Q'],self.hparams['Q'],self.hparams['h']))
        # d = torch.zeros((self.hparams['batch_size'],self.hparams['h']), dtype=self.hparams['dtype'], device=self.device)
        # d += 1/torch.sum(self.xi_Omega)*torch.einsum('Nijm,ij,Nij->Nm', Trunk_test, self.xi_Omega, f)
        # d += -1/torch.sum(self.xi_Gamma_b)*torch.einsum('Nijm,ij,Nij->Nm', Trunk_test, self.xi_Gamma_b, etab)
        # d += -1/torch.sum(self.xi_Gamma_t)*torch.einsum('Nijm,ij,Nij->Nm', Trunk_test, self.xi_Gamma_t, etat)
        d = 1/torch.sum(self.xi_Omega)*torch.sum(Trunk_test*self.xi_Omega[None,:,:,None]*f[:,:,:,None], axis=(1,2))
        d += -1/torch.sum(self.xi_Gamma_b)*torch.sum(Trunk_test*self.xi_Gamma_b[None,:,:,None]*etab[:,:,:,None], axis=(1,2))
        d += -1/torch.sum(self.xi_Gamma_t)*torch.sum(Trunk_test*self.xi_Gamma_t[None,:,:,None]*etat[:,:,:,None], axis=(1,2))  
        # d = F + Hb + Ht
        return d
        
    def forward_NGO(self, theta, f, etab, etat, x):
        K = self.compute_K(theta, x)
        d = self.compute_d(f, etab, etat, x)
        K_inv = self.NLBranch.forward(K)
        # K_inv = torch.linalg.inv(K)
        # u_coeff = torch.einsum('Nnm,Nm->Nn', K_inv, d)
        # u_coeff = torch.matmul(K_inv, d)
        u_coeff = torch.sum(K_inv[:,:,:]*d[:,None,:], axis=-1)
        Trunk_trial = self.Trunk_trial.forward(x)
        u_hat = torch.sum(u_coeff[:,None,:]*Trunk_trial, axis=-1)
        return u_hat
    
    def forward_VarMiON(self, theta, f, etab, etat, x):
        NLBranch = self.NLBranch.forward(theta)
        LBranch = self.LBranch_f.forward(f) + self.LBranch_eta.forward(etab*self.xi_Gamma_b + etat*self.xi_Gamma_t)
        latentvector = torch.einsum('nij,nj->ni', NLBranch, LBranch)
        Trunk = self.Trunk_trial.forward(x)
        u_hat = torch.einsum('ni,noi->no', latentvector, Trunk)
        return u_hat
    
    def forward(self, theta, f, etab, etat, x):
        if self.hparams.get('VarMiON',False)==True:
            u_hat = self.forward_VarMiON(theta, f, etab, etat, x)
        else:
            u_hat = self.forward_NGO(theta, f, etab, etat, x)
        return u_hat

    def symgroupavg_forward(self, theta, f, etab, etat, x):
        Theta_D8 = expand_D8(Theta)
        F_D8 = expand_D8(F)
        N_D8 = expand_D8(N)
        u_hat = torch.zeros((x.shape[0], x.shape[1]), device=self.device)
        for alpha in range(len(self.symgroup)):
            self.Trunk_trial.mapping = self.symgroup_inv[alpha].to(self.device)
            u_hat += self.forward(Theta_D8[alpha], F_D8[alpha], N_D8[alpha], x)
        u_hat = u_hat/len(self.symgroup)
        return u_hat
    
    def simforward(self, theta, f, etab, etat, x):
        theta = torch.tensor(theta, dtype=self.hparams['dtype']).tile((2,1,1))
        f = torch.tensor(f, dtype=self.hparams['dtype']).tile((2,1,1))
        etab = torch.tensor(etab, dtype=self.hparams['dtype']).tile((2,1,1))
        etat = torch.tensor(etat, dtype=self.hparams['dtype']).tile((2,1,1))
        x = torch.tensor(x, dtype=self.hparams['dtype']).tile((2,1,1))
        if self.hparams.get('symgroupavg',False)==True:    
            u = self.symgroupavg_forward(theta, f, etab, etat, x)
        else:
            u = self.forward(theta, f, etab, etat, x)
        u = u[0]
        u = torch.detach(u).cpu()
        u = np.array(u)
        return u

    def configure_optimizers(self):
        optimizer = self.hparams['optimizer'](self.parameters(), lr=self.hparams['learning_rate'])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        theta, f, etab, etat, x, u = train_batch
        if self.hparams.get('symgroupavg',False)==True:    
            u_hat = self.symgroupavg_forward(theta, f, etab, etat, x)
        else:
            u_hat = self.forward(theta, f, etab, etat, x)
        loss = 0
        for i in range(len(self.hparams['loss_coeffs'])):
            loss = loss + self.hparams['loss_coeffs'][i]*self.hparams['loss_terms'][i](u_hat, u)
        loss = loss/sum(self.hparams['loss_coeffs'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        theta, f, etab, etat, x, u = val_batch
        if self.hparams.get('symgroupavg',False)==True:    
            u_hat = self.symgroupavg_forward(theta, f, etab, etat, x)
        else:
            u_hat = self.forward(theta, f, etab, etat, x)
        loss = 0
        for i in range(len(self.hparams['loss_coeffs'])):
            loss = loss + self.hparams['loss_coeffs'][i]*self.hparams['loss_terms'][i](u_hat, u)
        loss = loss/sum(self.hparams['loss_coeffs']) 
        self.log('val_loss', loss)
        metric = self.hparams['metric'](u_hat, u)
        self.log('metric', metric)
        
    def geometry(self):
        #Domain
        self.xi_Omega = torch.ones((self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.device)
        x_0_Q, x_1_Q = np.mgrid[0:1:self.hparams['Q']*1j, 0:1:self.hparams['Q']*1j]
        x_Q = np.vstack([x_0_Q.ravel(), x_1_Q.ravel()]).T
        x_Q = np.tile(x_Q,(self.hparams['batch_size'],1,1))
        self.x_Q = torch.tensor(x_Q, dtype=self.hparams['dtype'], device=self.device)
        #Boundaries
        self.xi_Gamma_b = torch.zeros((self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.device)
        self.xi_Gamma_b[1:-1,0] = 1
        self.xi_Gamma_t = torch.zeros((self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.device)
        self.xi_Gamma_t[1:-1,-1] = 1
        self.xi_Gamma_l = torch.zeros((self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.device)
        self.xi_Gamma_l[0,1:-1] = 1
        self.xi_Gamma_r = torch.zeros((self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.device)
        self.xi_Gamma_r[-1,1:-1] = 1
        self.xi_Gamma = self.xi_Gamma_b + self.xi_Gamma_t + self.xi_Gamma_l + self.xi_Gamma_r
        self.xi_Gamma_eta = self.xi_Gamma_b + self.xi_Gamma_t
        self.xi_Gamma_g = self.xi_Gamma_l + self.xi_Gamma_r
        #Outward normal
        self.n = torch.zeros((self.hparams['Q'],self.hparams['Q'],self.params['simparams']['d']), dtype=self.hparams['dtype'], device=self.device)
        self.n[0,1:-1,:] = torch.tensor([-1,0], dtype=self.hparams['dtype'], device=self.device)
        self.n[-1,1:-1,:] = torch.tensor([1,0], dtype=self.hparams['dtype'], device=self.device)
        self.n[1:-1,0,:] = torch.tensor([0,-1], dtype=self.hparams['dtype'], device=self.device)
        self.n[1:-1,-1,:] = torch.tensor([0,1], dtype=self.hparams['dtype'], device=self.device)
    
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