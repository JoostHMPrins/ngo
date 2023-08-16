import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from customlayers import GaussianRBF, expand_D8

class RegularNN(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hparams.update(params['hparams'])
        self = self.to(self.hparams['dtype'])
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(434, 72, bias=self.hparams.get('bias_LBranch',True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(72, 1, bias=self.hparams.get('bias_LBranch',True)))

    def forward(self, Theta, F, N, x):
        Theta = Theta.flatten(-2,-1)
        F = F.flatten(-2,-1)
        N = N.flatten(-2,-1)
        y = torch.zeros(x.shape[0], x.shape[1], dtype=self.hparams['dtype']).to(self.device)
        for i in range(x.shape[1]):
            x_in = torch.cat((Theta, F, N, x[:,i,:]), dim=-1).to(self.device)
            for layer in self.layers:
                x_in = layer(x_in.float())
            y[:,i] = x_in.squeeze()
        return y
    
    def simforward(self, Theta, F, N, x):
        Theta = torch.tensor(Theta, dtype=self.hparams['dtype'])
        F = torch.tensor(F, dtype=self.hparams['dtype'])
        N = torch.tensor(N, dtype=self.hparams['dtype'])
        x = torch.tensor(x, dtype=self.hparams['dtype'])
        Theta = Theta.flatten(-2,-1)
        F = F.flatten(-2,-1)
        N = N.flatten(-2,-1)
        y = torch.zeros(x.shape[0], dtype=self.hparams['dtype']).to(self.device)
        for i in range(x.shape[0]):
            x_in = torch.cat((Theta, F, N, x[i,:]), dim=-1).to(self.device)
            for layer in self.layers:
                x_in = layer(x_in.float())
            y[i] = x_in
        return y 

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