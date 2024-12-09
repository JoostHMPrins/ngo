import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from numba import jit
import opt_einsum

from NeuralOperator import NeuralOperator
from darcy_ms import *

import sys
sys.path.insert(0, '../../ml') 
from quadrature import *
from basisfunctions import *

sys.path.insert(0, '../../trainingdata')
from datasaver import load_function_list


class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.device = self.hparams['used_device']
        self.data_dir = data_dir
        self.N_samples = self.hparams['N_samples_train'] + self.hparams['N_samples_val']
        dummymodel = NeuralOperator(self.hparams)
        if self.data_dir!=None:
            #Load input and output functions
            theta = load_function_list(variable='theta', loaddir=self.data_dir)
            f = load_function_list(variable='f', loaddir=self.data_dir)
            etab = load_function_list(variable='etab', loaddir=self.data_dir)
            etat = load_function_list(variable='etat', loaddir=self.data_dir)
            gl = load_function_list(variable='gl', loaddir=self.data_dir)
            gr = load_function_list(variable='gr', loaddir=self.data_dir)
            u = load_function_list(variable='u', loaddir=self.data_dir)
        else:
            # Generate input and output functions
            print('Generating functions...')
            dataset = ManufacturedSolutionsSetDarcy(N_samples=self.N_samples, variables=self.hparams['variables'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'], device=self.device)
            theta = dataset.theta
            f = dataset.f
            etab = dataset.etab
            etat = dataset.etat
            gl = dataset.gl
            gr = dataset.gr
            u = dataset.u
        #Discretize input functions
        print('Discretizing functions...')
        self.theta, self.theta_g, self.f, self.etab, self.etat, self.gl, self.gr = dummymodel.discretize_input_functions(theta, f, etab, etat, gl, gr)
        self.theta_bar = torch.sum(dummymodel.w_Omega[None,:]*self.theta, axis=-1)
        self.u = dummymodel.discretize_output_function(u)
        print('Assembling system...')
        if dummymodel.hparams['modeltype']=='model NGO' or dummymodel.hparams['modeltype']=='data NGO':
                self.F = dummymodel.compute_F(self.theta, self.theta_g)
                self.d = dummymodel.compute_d(self.f, self.etab, self.etat, self.gl, self.gr)

    def setup(self, stage=None):
        if self.hparams['modeltype']=='NN' or self.hparams['modeltype']=='DeepONet' or self.hparams['modeltype']=='VarMiON':
            self.theta = torch.tensor(self.theta, dtype=self.hparams['dtype'])
            self.f = torch.tensor(self.f, dtype=self.hparams['dtype'])
            self.etab = torch.tensor(self.etab, dtype=self.hparams['dtype'])
            self.etat = torch.tensor(self.etat, dtype=self.hparams['dtype'])
            self.gl = torch.tensor(self.gl, dtype=self.hparams['dtype'])
            self.gr = torch.tensor(self.gr, dtype=self.hparams['dtype']) 
            self.u = torch.tensor(self.u, dtype=self.hparams['dtype'])             
            dataset = torch.utils.data.TensorDataset(self.theta, self.f, self.etab, self.etat, self.gl, self.gr, self.u)
            # self.trainingset = torch.utils.data.TensorDataset(self.theta[:self.hparams['N_samples_train']], self.f[:self.hparams['N_samples_train']], self.etab[:self.hparams['N_samples_train']], self.etat[:self.hparams['N_samples_train']], self.gl[:self.hparams['N_samples_train']], self.gr[:self.hparams['N_samples_train']], self.u[:self.hparams['N_samples_train']])
            # self.validationset = torch.utils.data.TensorDataset(self.theta[self.hparams['N_samples_train']:], self.f[self.hparams['N_samples_train']:], self.etab[self.hparams['N_samples_train']:], self.etat[self.hparams['N_samples_train']:], self.gl[self.hparams['N_samples_train']:], self.gr[self.hparams['N_samples_train']:], self.u[self.hparams['N_samples_train']:])
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO' or self.hparams['modeltype']=='matrix data NGO':
            self.theta_bar = torch.tensor(self.theta_bar, dtype=self.hparams['dtype'])            
            self.F = torch.tensor(self.F, dtype=self.hparams['dtype'])
            self.d = torch.tensor(self.d, dtype=self.hparams['dtype'])
            self.u = torch.tensor(self.u, dtype=self.hparams['dtype'])   
            dataset = torch.utils.data.TensorDataset(self.theta_bar, self.F, self.d, self.u)
            # self.trainingset = torch.utils.data.TensorDataset(self.theta_bar[:self.hparams['N_samples_train']], self.F[:self.hparams['N_samples_train']], self.d[:self.hparams['N_samples_train']], self.u[:self.hparams['N_samples_train']])
            # self.validationset = torch.utils.data.TensorDataset(self.theta_bar[self.hparams['N_samples_train']:], self.F[self.hparams['N_samples_train']:], self.d[self.hparams['N_samples_train']:], self.u[self.hparams['N_samples_train']:])
        self.trainingset, self.validationset = random_split(dataset, [self.hparams['N_samples_train'], self.hparams['N_samples_val']])

    def train_dataloader(self):
        return DataLoader(self.trainingset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=0, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.validationset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=0, pin_memory=False)