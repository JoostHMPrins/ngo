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
sys.path.insert(0, '../../ml') 
from quadrature import *


class DataModule_Darcy_MS(pl.LightningDataModule):

    def __init__(self, data_dir, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.device = self.hparams['used_device']
        self.data_dir = data_dir
        dummymodel = NeuralOperator(self.hparams)
        # Generate input and output functions
        print('Generating functions...')
        dataset = ManufacturedSolutionsSetDarcy(N_samples=self.hparams['N_samples'], d=self.hparams['d'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'], device=self.device)
        #Discretize input functions
        print('Preprocessing data...')
        self.theta, self.theta_g, self.f, self.etab, self.etat, self.gl, self.gr = dummymodel.discretize_input_functions(dataset.theta, dataset.f, dataset.etab, dataset.etat, dataset.gl, dataset.gr)
        self.u = dummymodel.discretize_output_function(dataset.u)
        if dummymodel.hparams['modeltype']=='model NGO' or dummymodel.hparams['modeltype']=='data NGO' or dummymodel.hparams['modeltype']=='matrix data NGO':
            self.F = dummymodel.compute_F(self.theta, self.theta_g)
            self.d = dummymodel.compute_d(self.f, self.etab, self.etat, self.gl, self.gr)
            if self.hparams['N']!=self.hparams['N_F']:
                self.F = np.linalg.pinv(self.F)

    def setup(self, stage=None):
        if self.hparams['modeltype']=='NN' or self.hparams['modeltype']=='DeepONet' or self.hparams['modeltype']=='VarMiON' or self.hparams['modeltype']=='FNO':
            self.theta = torch.tensor(self.theta, dtype=self.hparams['dtype'])
            self.f = torch.tensor(self.f, dtype=self.hparams['dtype'])
            self.etab = torch.tensor(self.etab, dtype=self.hparams['dtype'])
            self.etat = torch.tensor(self.etat, dtype=self.hparams['dtype'])
            self.gl = torch.tensor(self.gl, dtype=self.hparams['dtype'])
            self.gr = torch.tensor(self.gr, dtype=self.hparams['dtype']) 
            self.u = torch.tensor(self.u, dtype=self.hparams['dtype'])             
            dataset = torch.utils.data.TensorDataset(self.theta, self.f, self.etab, self.etat, self.gl, self.gr, self.u)
            self.trainingset, self.validationset = random_split(dataset, [int(0.9*self.hparams['N_samples']), self.hparams['N_samples'] - int(0.9*self.hparams['N_samples'])])
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO' or self.hparams['modeltype']=='matrix data NGO':
            self.F = torch.tensor(self.F, dtype=self.hparams['dtype'])
            self.d = torch.tensor(self.d, dtype=self.hparams['dtype'])
            self.u = torch.tensor(self.u, dtype=self.hparams['dtype'])   
            dataset = torch.utils.data.TensorDataset(self.F, self.d, self.u)
            self.trainingset, self.validationset = random_split(dataset, [int(0.9*self.hparams['N_samples']), self.hparams['N_samples'] - int(0.9*self.hparams['N_samples'])])

    def train_dataloader(self):
        return DataLoader(self.trainingset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=0, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.validationset, batch_size=self.hparams['N_samples']-int(0.9*self.hparams['N_samples']), shuffle=False, num_workers=0, pin_memory=False)