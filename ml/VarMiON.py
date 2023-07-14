import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from torch_rbf import RBF, gaussian


class GaussianRBF(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #Definition and initialization of centers and scales
        self.mus = torch.nn.Parameter(torch.ones(output_dim, input_dim))
        self.log_sigmas = torch.nn.Parameter(torch.ones(output_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.mus, 0, 1)
        nn.init.constant_(self.log_sigmas, -1)
        
    def forward(self, x):
        d_scaled = ((x[:,:,None,:] - self.mus[None,None,:,:])/torch.exp(self.log_sigmas[None,None,:,None]))
        output = torch.exp(-1/2*(torch.linalg.vector_norm(d_scaled, axis=-1, ord=2))**2)
        return torch.exp(-1/2*(torch.linalg.vector_norm(d_scaled, axis=-1, ord=2))**2)
    
    
class NLBranchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=4, stride=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=16))
        self.layers.append(nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=4, stride=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=32))
        self.layers.append(nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2))
        self.layers.append(nn.ReLU())

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        x = x[0]
        return x

    
class LBranchNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class TrunkNet(nn.Module):
    def __init__(self, input_dim, output_dim, basis_func):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(RBF(in_features=input_dim, out_features=output_dim, basis_func=basis_func))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class VarMiON(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hparams.update(params['hparams'])
        self.NLBranch = NLBranchNet()
        self.LBranchF = LBranchNet(input_dim=144, output_dim=72)
        self.LBranchN = LBranchNet(input_dim=144, output_dim=72)
        # self.Trunk = TrunkNet(input_dim=2, output_dim=72, basis_func=gaussian)
        # self.Trunk = RBF(2,72,gaussian)
        self.Trunk = GaussianRBF(2,72)
        
    def forward(self, Theta, F, N, x):
        NLBranch = self.NLBranch.forward(Theta)
        LBranch = self.LBranchF.forward(F) + self.LBranchN.forward(N)
        Branch = torch.einsum('nij,nj->ni', NLBranch, LBranch)
        Trunk = self.Trunk.forward(x)
        u_hat = torch.einsum('ni,noi->no', Branch, Trunk)
        return u_hat
    
    def simforward(self, Theta, F, N, x):
        Theta = torch.tensor(Theta, dtype=self.hparams['dtype'])
        Theta = Theta.unsqueeze(0)
        F = torch.tensor(F, dtype=self.hparams['dtype'])
        F = F.unsqueeze(0)
        N = torch.tensor(N, dtype=self.hparams['dtype'])
        N = N.unsqueeze(0)
        x = torch.tensor(x, dtype=self.hparams['dtype'])
        x = x.unsqueeze(0)
        u = self.forward(Theta, F, N, x)
        u = torch.detach(u).cpu()
        u = np.array(u)
        u = u[0]
        return u

    def configure_optimizers(self):
        optimizer = self.hparams['optimizer'](self.parameters(), lr=self.hparams['learning_rate'])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        Theta, F, N, x, u = train_batch
        u_hat = self.forward(Theta, F, N, x)
        eps = 2**(-52)
        loss = nn.functional.mse_loss(u_hat, u)
        # loss = 0
        # for i in range(len(self.hparams['loss_coeffs'])):
        #     loss = loss + self.hparams['loss_coeffs'][i]*self.hparams['loss_terms'][i](self, u_hat, u)
        # loss = loss/sum(self.hparams['loss_coeffs'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        Theta, F, N, x, u = val_batch
        u_hat = self.forward(Theta, F, N, x)
        eps = 2**(-52)
        loss = nn.functional.mse_loss(u_hat, u)
        # loss = 0
        # for i in range(len(self.hparams['loss_coeffs'])):
        #     loss = loss + self.hparams['loss_coeffs'][i]*self.hparams['loss_terms'][i](self, u_hat, u)
        # loss = loss/sum(self.hparams['loss_coeffs'])
        self.log('val_loss', loss)
        metric = self.hparams['metric'](u_hat, u)
        self.log('metric', metric)
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint['params'] = self.params
