import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
    
    
class DataModule_hc2d(pl.LightningDataModule):

    def __init__(self, data_dir, hparams):
        super().__init__()
        
        self.data_dir = data_dir
        self.hyperparams = hparams

    def setup(self, stage=None):
        self.Theta = np.load(self.data_dir + '/Theta.npy')
        self.F = np.load(self.data_dir + '/F.npy')
        self.N = np.load(self.data_dir + '/N.npy')
        self.x = np.load(self.data_dir + '/x.npy')
        self.u = np.load(self.data_dir + '/u.npy')
        self.Theta = torch.tensor(self.Theta, dtype=self.hyperparams['dtype'])
        self.F = torch.tensor(self.F, dtype=self.hyperparams['dtype'])
        self.N = torch.tensor(self.N, dtype=self.hyperparams['dtype'])
        self.x = torch.tensor(self.x, dtype=self.hyperparams['dtype'])
        self.u = torch.tensor(self.u, dtype=self.hyperparams['dtype'])
        
        self.dataset = torch.utils.data.TensorDataset(self.Theta, self.F, self.N, self.x, self.u)
        self.trainingset, self.validationset = random_split(self.dataset, [int(0.9*self.u.shape[0]), int(0.1*self.u.shape[0])])

    def train_dataloader(self):
        return DataLoader(self.trainingset, batch_size=self.hyperparams['batch_size'])

    def val_dataloader(self):
        return DataLoader(self.validationset, batch_size=self.hyperparams['batch_size'])
    
    
class DataModule_bm(pl.LightningDataModule):

    def __init__(self, data_dir, hparams):
        super().__init__()
        
        self.data_dir = data_dir
        self.hyperparams = hparams

    def setup(self, stage=None):
        self.x_in = np.load(self.data_dir + '/x_in.npy')
        self.x_D = np.load(self.data_dir + '/x_D.npy')
        self.xi = np.load(self.data_dir + '/xi.npy')
        self.x_out = np.load(self.data_dir + '/x_out.npy')
        self.x_in = torch.tensor(self.x_in, dtype=self.hyperparams['dtype'])
        self.x_D = torch.tensor(self.x_D, dtype=self.hyperparams['dtype'])
        self.xi = torch.tensor(self.xi, dtype=self.hyperparams['dtype'])
        self.x_out = torch.tensor(self.x_out, dtype=self.hyperparams['dtype'])
        
        self.dataset = torch.utils.data.TensorDataset(self.x_in, self.x_D, self.xi, self.x_out)
        self.trainingset, self.validationset = random_split(self.dataset, [int(0.9*self.x_out.shape[0]), int(0.1*self.x_out.shape[0])])

    def train_dataloader(self):
        return DataLoader(self.trainingset, batch_size=self.hyperparams['batch_size'])

    def val_dataloader(self):
        return DataLoader(self.validationset, batch_size=self.hyperparams['batch_size'])