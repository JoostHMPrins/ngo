import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F

class MLP(pl.LightningModule):
    def __init__(self, trainingdataparams, hparams, label):
        super().__init__()
        self.label = label
        self.hparams.update(hparams)
        self.trainingdataparams = trainingdataparams
        self.lr = self.hparams['learning_rate']
        #Network structure
        current_dim = self.hparams['input_dim']
        self.layers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(len(self.hparams['hidden_dim'])):
            self.layers.append(nn.Linear(current_dim, self.hparams['hidden_dim'][i], bias=self.hparams['bias']))
            #print(self.layers[i])
            self.activations.append(self.hparams['activations'][i])
            #print(self.activations[i])
            if self.hparams['batchnorm']==True:
                self.batchnorms.append(nn.BatchNorm1d(self.hparams['hidden_dim'][i]))
                #print(self.batchnorms[i])
            current_dim = self.hparams['hidden_dim'][i]
        self.layers.append(nn.Linear(current_dim, self.hparams['output_dim'], bias=self.hparams['bias']))
        #print(self.layers[-1])
        if self.hparams['output_activation']==True:
            self.activations.append(self.hparams['activations'][-1])
            #print(self.activations[-1])
            
    def forward(self, x):
        x_0 = np.copy(x)
        for i in range(len(self.hparams['hidden_dim'])):
            x = self.layers[i](x)
            x = self.activations[i](x) + x if self.hparams['resnet']==True else self.activations[i](x)
            if self.hparams.get('batchnorm',False)==True:
                x = self.batchnorms[i](x)
        x = self.layers[-1](x)
        if self.hparams['skipconnection']==True:
            Deltax = np.copy(x)
            y = x_0 + Deltax
        else:
            y = x
        if self.hparams['output_activation']==True:
            y = self.activations[-1](y)
        return y 