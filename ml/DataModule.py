import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from numba import jit
import opt_einsum

import sys
sys.path.insert(0, '/home/prins/st8/prins/phd/gitlab/ngo-pde-gk/fem') 
from datasaver import load_function_list

from NGO import NGO
    

class DataModule_hc2d(pl.LightningDataModule):

    def __init__(self, data_dir, params):
        super().__init__()
        
        self.params = params
        self.data_dir = data_dir
        self.hyperparams = params['hparams']
    
    def setup(self, stage=None):
        dummyNGO = NGO(self.params)
        # Load input and output functions
        print('Preprocessing data...')
        theta_raw = load_function_list(variable='theta', loaddir=self.data_dir)
        f_raw = load_function_list(variable='f', loaddir=self.data_dir)
        etat_raw = load_function_list(variable='etat', loaddir=self.data_dir)
        etab_raw = load_function_list(variable='etab', loaddir=self.data_dir)
        x_raw = np.load(self.data_dir + '/x.npy')
        u_raw = np.load(self.data_dir + '/u.npy')
        #Discretize input functions
        self.theta, self.theta_g, self.f, self.etab, self.etat = dummyNGO.discretize_input_functions(theta_raw, f_raw, etab_raw, etat_raw)
        self.K = torch.tensor(dummyNGO.compute_K(self.theta, self.theta_g), dtype=self.hyperparams['dtype'])
        self.d = torch.tensor(dummyNGO.compute_d(self.f, self.etab, self.etat), dtype=self.hyperparams['dtype'])
        psi = dummyNGO.Trunk_trial.forward(x_raw.reshape((x_raw.shape[0]*x_raw.shape[1],x_raw.shape[2]))).reshape((x_raw.shape[0],x_raw.shape[1],self.hyperparams['h']))
        #Sample x, psi and u
        self.x = []
        self.psi = []
        self.u = []
        for i in range(len(theta_raw)):
            indices = np.linspace(0,x_raw.shape[1]-1, x_raw.shape[1], dtype=int)
            indices_output = np.random.choice(indices, size=self.hyperparams['Q_L'], replace=False)
            self.x.append(x_raw[i,indices_output])
            self.psi.append(psi[i,indices_output])
            self.u.append(u_raw[i,indices_output])
        self.x = torch.tensor(np.array(self.x), dtype=self.hyperparams['dtype'])
        self.psi = torch.tensor(np.array(self.psi), dtype=self.hyperparams['dtype'])
        self.u = torch.tensor(np.array(self.u), dtype=self.hyperparams['dtype'])
        #Define dataset
        self.dataset = torch.utils.data.TensorDataset(self.theta, self.f, self.etab, self.etat, self.K, self.d, self.psi, self.x, self.u)
        self.trainingset, self.validationset = random_split(self.dataset, [int(0.9*self.u.shape[0]), int(0.1*self.u.shape[0])])

    def train_dataloader(self):
        return DataLoader(self.trainingset, batch_size=self.hyperparams['batch_size'])#, num_workers=10, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validationset, batch_size=self.hyperparams['batch_size'])#, num_workers=10, pin_memory=True)


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