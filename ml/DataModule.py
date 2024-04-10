import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl

import sys
sys.path.insert(0, '/home/prins/st8/prins/phd/gitlab/ngo-pde-gk/fem') 
from datasaver import load_function_list

from NGO import NGO
    

class DataModule_hc2d(pl.LightningDataModule):

    def __init__(self, data_dir, hparams):
        super().__init__()
        
        self.data_dir = data_dir
        self.hyperparams = hparams

    def setup(self, stage=None):
        #Load and discretize input and output functions
        print('Preprocessing data...')
        self.theta_raw = load_function_list(variable='theta', loaddir=self.data_dir)
        self.f_raw = load_function_list(variable='f', loaddir=self.data_dir)
        self.etat_raw = load_function_list(variable='etat', loaddir=self.data_dir)
        self.etab_raw = load_function_list(variable='etab', loaddir=self.data_dir)
        self.x_raw = np.load(self.data_dir + '/x.npy')
        self.u_raw = np.load(self.data_dir + '/u.npy')
        #Define quadrature grid
        x_0, x_1 = np.mgrid[0:1:self.hyperparams['Q']*1j, 0:1:self.hyperparams['Q']*1j]
        x_Q = np.vstack([x_0.ravel(), x_1.ravel()]).T
        #Define empty lists for data
        self.theta = []
        self.f = []
        self.etab = []
        self.etat = []
        self.x = []
        self.u = []
        #Discretize input functions and sample output data
        for i in range(len(self.theta_raw)):
            self.theta.append(self.theta_raw[i](x_Q))
            self.f.append(self.f_raw[i](x_Q))
            self.etab.append(self.etab_raw[i](x_Q))
            self.etat.append(self.etat_raw[i](x_Q))
            indices = np.linspace(0,self.x_raw.shape[1]-1, self.x_raw.shape[1], dtype=int)
            indices_output = np.random.choice(indices, size=self.hyperparams['Q_L'], replace=False)
            self.x.append(self.x_raw[i,indices_output])
            self.u.append(self.u_raw[i,indices_output])
        #Convert to numpy arrays first (faster)   
        self.theta = np.array(self.theta)
        self.f = np.array(self.f)
        self.etab = np.array(self.etab)
        self.etat = np.array(self.etat)
        self.x = np.array(self.x)
        self.u = np.array(self.u)
        #Convert to torch tensors
        self.theta = torch.tensor(self.theta, dtype=self.hyperparams['dtype']).reshape(len(self.theta_raw),self.hyperparams['Q'],self.hyperparams['Q'])
        self.f = torch.tensor(self.f, dtype=self.hyperparams['dtype']).reshape(len(self.theta_raw),self.hyperparams['Q'],self.hyperparams['Q'])
        self.etab = torch.tensor(self.etab, dtype=self.hyperparams['dtype']).reshape(len(self.theta_raw),self.hyperparams['Q'],self.hyperparams['Q'])
        self.etat = torch.tensor(self.etat, dtype=self.hyperparams['dtype']).reshape(len(self.theta_raw),self.hyperparams['Q'],self.hyperparams['Q'])
        self.x = torch.tensor(self.x, dtype=self.hyperparams['dtype'])
        self.u = torch.tensor(self.u, dtype=self.hyperparams['dtype'])
        
        self.dataset = torch.utils.data.TensorDataset(self.theta, self.f, self.etab, self.etat, self.x, self.u)
        self.trainingset, self.validationset = random_split(self.dataset, [int(0.9*self.u.shape[0]), int(0.1*self.u.shape[0])])

    def train_dataloader(self):
        return DataLoader(self.trainingset, batch_size=self.hyperparams['batch_size'], num_workers=10, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validationset, batch_size=self.hyperparams['batch_size'], num_workers=10, pin_memory=True)


class DataModule_hc2d_old(pl.LightningDataModule):

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