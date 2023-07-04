import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class GaussianRBF(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        #Definition and initialization of centers and scales
        self.mus = torch.nn.Parameter(torch.ones(output_dim, input_dim))
        self.log_sigmas = torch.nn.Parameter(torch.ones(output_dim))
        nn.init.uniform_(self.mus, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)
        
    def forward(self, x):
        d_scaled = (x[:,None,:] - self.mus[None,:,:]/torch.exp(self.log_sigmas[None,:,None]))
        return torch.exp(-(torch.linalg.vector_norm(d_scaled, axis=-1, ord=2))**2/2)
    
    
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
        for layer in self.layers:
            x = layer(x)
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
    
    
class VarMiON(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.hparams.update(params['hparams'])

    def forward(self, Theta, F, N, x):
        NLBranch = NLBranchNet().forward(Theta)
        NLBranch = NLBranch.reshape((NLBranch.shape[0], NLBranch.shape[2], NLBranch.shape[3]))
        LBranch = LBranchNet(144,72).forward(F) + LBranchNet(144,72).forward(N)
        Branch = torch.einsum('nij,nj->ni', NLBranch, LBranch)
        Trunk = GaussianRBF(2,72).forward(x)
        u_hat = torch.einsum('ni,ni->n', Branch, Trunk)
        return u_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        Theta, F, N, x, u = train_batch
        u_hat = VarMiON().forward(Theta, F, N, x)
        loss = F.mse_loss(u_hat, u)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        Theta, F, N, x, u = val_batch
        u_hat = VarMiON().forward(Theta, F, N, x)
        loss = F.mse_loss(u_hat, u)
        self.log('val_loss', loss)