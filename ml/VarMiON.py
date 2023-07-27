import torch
from torch import nn
import pytorch_lightning as pl
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
        nn.init.constant_(self.log_sigmas, np.log(0.15))
        
    def forward(self, x):
        d_scaled = (x[:,:,None,:] - self.mus[None,None,:,:])/torch.exp(self.log_sigmas[None,None,:,None])
        y = torch.exp(-torch.sum(d_scaled**2, axis=-1))
        if self.hparams.get('norm_basis',False)==True:
            y = y/torch.sum(y, axis=-1)[:,:,None]
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
        if self.hparams.get('NLB_outputReLU',False)==True:
            self.layers.append(nn.ReLU())

    def forward(self, x):
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
    
    
class VarMiON(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hparams.update(params['hparams'])
        self.NLBranch = NLBranchNet(params)
        self.LBranchF = LBranchNet(params, input_dim=144, output_dim=72)
        self.LBranchN = LBranchNet(params, input_dim=144, output_dim=72)
        self.Trunk = GaussianRBF(params, input_dim=2, output_dim=72)
        self = self.to(self.hparams['dtype'])
        
    def forward(self, Theta, F, N, x):
        NLBranch = self.NLBranch.forward(Theta)
        LBranch = self.LBranchF.forward(F) + self.LBranchN.forward(N)
        Branch = torch.einsum('nij,nj->ni', NLBranch, LBranch)
        Trunk = self.Trunk.forward(x)
        u_hat = torch.einsum('ni,noi->no', Branch, Trunk)
        return u_hat           
    
    def simforward(self, Theta, F, N, x):
        Theta = torch.tensor(Theta, dtype=self.hparams['dtype'])
        F = torch.tensor(F, dtype=self.hparams['dtype'])
        N = torch.tensor(N, dtype=self.hparams['dtype'])
        x = torch.tensor(x, dtype=self.hparams['dtype'])
        Theta = Theta.unsqueeze(0)
        F = F.unsqueeze(0)
        N = N.unsqueeze(0)
        x = x.unsqueeze(1).unsqueeze(1)
        NLBranch = self.NLBranch.forward(Theta).squeeze()
        LBranch = self.LBranchF.forward(F).squeeze() + self.LBranchN.forward(N).squeeze()
        Branch = torch.einsum('ij,j->i', NLBranch, LBranch)
        Trunk = self.Trunk.forward(x).squeeze()
        u = torch.einsum('i,oi->o', Branch, Trunk)
        u = torch.detach(u).cpu()
        u = np.array(u)
        return u

    def configure_optimizers(self):
        optimizer = self.hparams['optimizer'](self.parameters(), lr=self.hparams['learning_rate'])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        Theta, F, N, x, u = train_batch
        u_hat = self.forward(Theta, F, N, x)
        loss = 0
        for i in range(len(self.hparams['loss_coeffs'])):
            loss = loss + self.hparams['loss_coeffs'][i]*self.hparams['loss_terms'][i](u_hat, u)
        loss = loss/sum(self.hparams['loss_coeffs'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        Theta, F, N, x, u = val_batch
        u_hat = self.forward(Theta, F, N, x)
        loss = 0
        for i in range(len(self.hparams['loss_coeffs'])):
            loss = loss + self.hparams['loss_coeffs'][i]*self.hparams['loss_terms'][i](u_hat, u)
        loss = loss/sum(self.hparams['loss_coeffs'])
        self.log('val_loss', loss)
        metric = self.hparams['metric'](u_hat, u)
        self.log('metric', metric)
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint['params'] = self.params