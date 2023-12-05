import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from customlayers import *
from customlosses import *
    
class NLBranchNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        # self.layers.append(ResizeLayer2D(params, input_dim=self.hparams['input_dim'], output_dim=12))
        # self.layers.append(UnsqueezeLayer(params))
        self.layers.append(nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch',True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=16))
        self.layers.append(nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch',True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=32))
        self.layers.append(nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch',True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch',True)))
        # self.layers.append(SqueezeLayer(params))
        # self.layers.append(ResizeLayer2D(params, input_dim=72, output_dim=self.hparams['latent_dim']))
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
    
    
class HyperNet(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, output_dim, bias=self.hparams.get('bias_HyperNet', True)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        y = x
        return y
    
    
class VarMiON(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hparams.update(params['hparams'])
        self.NLBranch = NLBranchNet(params)
        self.LBranchF = LBranchNet(params, input_dim=144, output_dim=72)
        self.LBranchN = LBranchNet(params, input_dim=144, output_dim=72)
        if self.hparams.get('NOMAD',False)==True:
            self.HyperNet = HyperNet(params, input_dim=72, output_dim=3*72)
            self.Trunk = GaussianRBF_NOMAD(params, input_dim=params['simparams']['d'], output_dim=72)
        else:
            self.Trunk = GaussianRBF(params, input_dim=params['simparams']['d'], output_dim=72)
        self.compute_symgroup()
        self = self.to(self.hparams['dtype'])
        
    def forward(self, Theta, F, N, x):
        NLBranch = self.NLBranch.forward(Theta)
        LBranch = self.LBranchF.forward(F) + self.LBranchN.forward(N)
        latentvector = torch.einsum('nij,nj->ni', NLBranch, LBranch)
        if self.hparams.get('NOMAD',False)==True:
            mus_log_sigmas = self.HyperNet.forward(latentvector)
            mus = mus_log_sigmas[:,:2*72].reshape((mus_log_sigmas.shape[0],72,2))
            mus = torch.sigmoid(mus)
            log_sigmas = mus_log_sigmas[:,2*72:]
            Trunk = self.Trunk.forward(mus, log_sigmas, x)
            self.mus = mus
            self.log_sigmas = log_sigmas
        else:
            Trunk = self.Trunk.forward(x)
        u_hat = torch.einsum('ni,noi->no', latentvector, Trunk)
        # u_hat = torch.sum(Trunk, axis=-1)
        return u_hat           
    
    def symgroupavg_forward(self, Theta, F, N, x):
        Theta_D8 = expand_D8(Theta)
        F_D8 = expand_D8(F)
        N_D8 = expand_D8(N)
        u_hat = torch.zeros((x.shape[0], x.shape[1]), device=self.device)
        for alpha in range(len(self.symgroup)):
            self.Trunk.mapping = self.symgroup_inv[alpha].to(self.device)
            u_hat += self.forward(Theta_D8[alpha], F_D8[alpha], N_D8[alpha], x)
        u_hat = u_hat/len(self.symgroup)
        return u_hat
    
    def simforward(self, Theta, F, N, x):
        Theta = torch.tensor(Theta, dtype=self.hparams['dtype']).tile((2,1,1))
        F = torch.tensor(F, dtype=self.hparams['dtype']).tile((2,1,1))
        N = torch.tensor(N, dtype=self.hparams['dtype']).tile((2,1,1))
        x = torch.tensor(x, dtype=self.hparams['dtype']).tile((2,1,1))
        if self.hparams.get('symgroupavg',False)==True:    
            u = self.symgroupavg_forward(Theta, F, N, x)
        else:
            u = self.forward(Theta, F, N, x)
        u = u[0]
        u = torch.detach(u).cpu()
        u = np.array(u)
        return u

    def configure_optimizers(self):
        optimizer = self.hparams['optimizer'](self.parameters(), lr=self.hparams['learning_rate'])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        Theta, F, N, x, u = train_batch
        if self.hparams.get('symgroupavg',False)==True:    
            u_hat = self.symgroupavg_forward(Theta, F, N, x)
        else:
            u_hat = self.forward(Theta, F, N, x)
        loss = 0
        for i in range(len(self.hparams['loss_coeffs'])):
            loss = loss + self.hparams['loss_coeffs'][i]*self.hparams['loss_terms'][i](u_hat, u)
        loss = loss/sum(self.hparams['loss_coeffs'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        Theta, F, N, x, u = val_batch
        if self.hparams.get('symgroupavg',False)==True:    
            u_hat = self.symgroupavg_forward(Theta, F, N, x)
        else:
            u_hat = self.forward(Theta, F, N, x)
        loss = 0
        for i in range(len(self.hparams['loss_coeffs'])):
            loss = loss + self.hparams['loss_coeffs'][i]*self.hparams['loss_terms'][i](u_hat, u)
        loss = loss/sum(self.hparams['loss_coeffs'])
        self.log('val_loss', loss)
        metric = self.hparams['metric'](u_hat, u)
        self.log('metric', metric)
        
    def compute_symgroup(self):
        R = torch.tensor([[0,-1],[1,0]], dtype=self.hparams['dtype'], device=self.device)
        M = torch.tensor([[1,0],[0,-1]], dtype=self.hparams['dtype'], device=self.device)
        I = torch.tensor([[1,0],[0,1]], dtype=self.hparams['dtype'], device=self.device)
        # self.symgroup = [I, R, R@R, R@R@R, M, R@M, R@R@M, R@R@R@M]
        self.symgroup = [I, R@R, M, R@R@M]
        # self.symgroup_inv =[I, torch.linalg.inv(R), torch.linalg.inv(R@R), torch.linalg.inv(R@R@R), torch.linalg.inv(M), torch.linalg.inv(R@M), torch.linalg.inv(R@R@M), torch.linalg.inv(R@R@R@M)]
        self.symgroup_inv =[I, torch.linalg.inv(R@R), torch.linalg.inv(M), torch.linalg.inv(R@R@M)]
        
    def on_before_zero_grad(self, optimizer):
        if self.hparams.get('bound_mus',False)==True:
            for name, p in self.Trunk.named_parameters():
                if name=='mus':
                    p.data.clamp_(0, 1.0)
            
    def on_save_checkpoint(self, checkpoint):
        checkpoint['params'] = self.params