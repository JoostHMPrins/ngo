import numpy as np
import torch
import math
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
    
    
class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir, params):
        super().__init__()
        
        self.data_dir = data_dir
        
        with open(self.data_dir + '/params.json', 'r') as fp:
            params = json.load(fp)
            self.params = params

    def setup(self, stage=None):
        self.theta = np.load(self.data_dir + '/theta.npy')
        self.f = np.load(self.data_dir + '/f.npy')
        self.eta = np.load(self.data_dir + '/eta.npy')
        self.x = np.load(self.data_dir + '/x.npy')
        self.u = np.load(self.data_dir + '/u.npy')
        self.theta = torch.tensor(self.theta, dtype=self.hparams['dtype'])
        self.f = torch.tensor(self.f, dtype=self.hparams['dtype'])
        self.eta = torch.tensor(self.eta, dtype=self.hparams['dtype'])
        self.x = torch.tensor(self.x, dtype=self.hparams['dtype'])
        self.u = torch.tensor(self.u, dtype=self.hparams['dtype'])
        
        self.dataset = torch.utils.data.TensorDataset(self.f, self.theta, self.eta, self.x, self.u)
        self.trainingset, self.validationset = random_split(self.dataset, [int(0.9*self.u.shape[0]), int(0.1*self.u.shape[0])])

    def train_dataloader(self):
        return DataLoader(self.trainingset, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return DataLoader(self.validationset, batch_size=self.hparams['batch_size'])

    def test_dataloader(self):
        return DataLoader(self.validationset, batch_size=self.hparams['batch_size'])