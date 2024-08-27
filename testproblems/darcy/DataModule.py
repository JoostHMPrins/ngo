import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from numba import jit
import opt_einsum

from NeuralOperator import NeuralOperator
from darcy_mfs import *

import sys
sys.path.insert(0, '../../trainingdata') 
from datasaver import load_function_list
sys.path.insert(0, '../../ml') 
from quadrature import *


class DataModule_Darcy_MS(pl.LightningDataModule):

    def __init__(self, data_dir, params):
        super().__init__()
        
        self.params = params
        self.data_dir = data_dir
        self.hyperparams = params['hparams']
        
    def setup(self, stage=None):
        dummymodel = NeuralOperator(self.params)
        # Generate input and output functions
        print('Generating data...')
        dataset = ManufacturedSolutionsSetDarcy(N_samples=self.hyperparams['N_samples'], d=self.hyperparams['d'], l_min=self.hyperparams['l_min'], l_max=self.hyperparams['l_max'])
        #Discretize input functions
        print('Preprocessing data...')
        theta, theta_g, f, etab, etat, gl, gr = dummymodel.discretize_input_functions(dataset.theta, dataset.f, dataset.etab, dataset.etat, dataset.gl, dataset.gr)
        u = dummymodel.discretize_output_function(dataset.u)
        if dummymodel.hparams['modeltype']=='model NGO' or dummymodel.hparams['modeltype']=='data NGO' or dummymodel.hparams['modeltype']=='matrix data NGO':
            F = dummymodel.compute_F(theta, theta_g)
            d = dummymodel.compute_d(f, etab, etat, gl, gr)
            F = torch.tensor(F, dtype=self.hyperparams['dtype'])
            if self.hyperparams['N']!=self.hyperparams['N_F']:
                F = torch.linalg.pinv(F)
            d = torch.tensor(d, dtype=self.hyperparams['dtype'])
        #Convert to torch tensors
        theta = torch.tensor(theta, dtype=self.hyperparams['dtype'])
        theta_g = torch.tensor(theta_g, dtype=self.hyperparams['dtype'])
        f = torch.tensor(f, dtype=self.hyperparams['dtype'])
        etab = torch.tensor(etab, dtype=self.hyperparams['dtype'])
        etat = torch.tensor(etat, dtype=self.hyperparams['dtype'])
        gl = torch.tensor(gl, dtype=self.hyperparams['dtype'])
        gr = torch.tensor(gr, dtype=self.hyperparams['dtype'])
        u = torch.tensor(u, dtype=self.hyperparams['dtype'])
        #Define dataset
        if dummymodel.hparams['modeltype']=='DeepONet' or dummymodel.hparams['modeltype']=='VarMiON' or dummymodel.hparams['modeltype']=='FNO':
            dataset = torch.utils.data.TensorDataset(theta, f, etab, etat, gl, gr, u)
            self.trainingset, self.validationset = random_split(dataset, [int(0.9*self.hyperparams['N_samples']), self.hyperparams['N_samples'] - int(0.9*self.hyperparams['N_samples'])])
        if dummymodel.hparams['modeltype']=='NGO':
            dataset = torch.utils.data.TensorDataset(F, d, u)
            self.trainingset, self.validationset = random_split(dataset, [int(0.9*self.hyperparams['N_samples']), self.hyperparams['N_samples'] - int(0.9*self.hyperparams['N_samples'])])

    def train_dataloader(self):
        return DataLoader(self.trainingset, batch_size=self.hyperparams['batch_size'], shuffle=False, num_workers=2, pin_memory=False)#, persistent_workers=False)#, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.validationset, batch_size=self.hyperparams['N_samples']-int(0.9*self.hyperparams['N_samples']), shuffle=False, num_workers=2, pin_memory=False)#, persistent_workers=False)#, prefetch_factor=2)