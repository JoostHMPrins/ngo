
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from numba import jit
import opt_einsum

from NeuralOperator import NeuralOperator
from manufacturedsolutions import *

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
        self.hparams['used_device'] = self.hparams['assembly_device']
        self.data_dir = data_dir
        self.N_samples = self.hparams['N_samples_train'] + self.hparams['N_samples_val']
        dummymodel = NeuralOperator(self.hparams)
        # Generate input and output functions
        print('Generating functions...')
        dataset = ManufacturedSolutionsSet(N_samples=self.N_samples, variables=self.hparams['variables'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'], device=hparams['discretization_device'])
        theta = dataset.theta
        f = dataset.f
        eta_y0 = dataset.etab
        eta_yL = dataset.etat
        g_x0 = dataset.gl
        g_xL = dataset.gr
        u = dataset.u
        u0 = dataset.u0
        #Discretize input functions
        if self.hparams['modeltype']=='NN' or self.hparams['modeltype']=='DeepONet' or self.hparams['modeltype']=='VarMiON':
            print('Discretizing input functions...')
            self.theta_d, self.theta_x0_d, self.theta_xL_d, self.f_d, self.eta_y0_d, self.eta_yL_d, self.g_x0_d, self.g_xL_d, self.u0_d = dummymodel.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
        if dummymodel.hparams['modeltype']=='model NGO':
            print('Assembling F...')
            print('Assembling d...')
            print('Calculating conserved quantity...')
            bs = self.hparams['assembly_batch_size']
            n_batches = int(self.N_samples/bs)
            self.F = np.zeros((len(theta),self.hparams['N'],self.hparams['N']))
            self.d = np.zeros((len(f),self.hparams['N']))
            self.C = np.zeros(len(f))
            self.C_m = np.zeros((len(theta),self.hparams['N']))
            for i in range(n_batches):
                print('batch: '+str(i))
                self.F[bs*i:bs*(i+1)] = dummymodel.compute_F(theta[bs*i:bs*(i+1)])
                self.d[bs*i:bs*(i+1)] = dummymodel.compute_d(f[bs*i:bs*(i+1)], eta_y0[bs*i:bs*(i+1)], eta_yL[bs*i:bs*(i+1)], g_x0[bs*i:bs*(i+1)], g_xL[bs*i:bs*(i+1)], u0[bs*i:bs*(i+1)])
                self.C[bs*i:bs*(i+1)] = dummymodel.compute_C(f[bs*i:bs*(i+1)], eta_y0[bs*i:bs*(i+1)], eta_yL[bs*i:bs*(i+1)], u0[bs*i:bs*(i+1)])
                self.C_m[bs*i:bs*(i+1)] = dummymodel.compute_C_m(theta[bs*i:bs*(i+1)])
        if dummymodel.hparams['modeltype']=='data NGO':
            print('Assembling F...')
            self.F = dummymodel.compute_F(theta)
            print('Assembling d...')
            self.d = dummymodel.compute_d(f, eta_y0, eta_yL, g_x0, g_xL, u0)
            print('Calculating conserved quantity...')
            self.C = dummymodel.compute_C(f, eta_y0, eta_yL, u0)
            self.C_m = dummymodel.compute_C_m(theta)
        print('Discretizing output function...')
        if dummymodel.hparams['project_u']==True:
            self.u_d = dummymodel.project_output_function(u)
        if dummymodel.hparams['project_u']==False:
            self.u_d = dummymodel.discretize_output_function(u)    
        
    def setup(self, stage=None):
        if self.hparams['modeltype']=='NN' or self.hparams['modeltype']=='DeepONet' or self.hparams['modeltype']=='VarMiON':
            self.theta = torch.tensor(self.theta_d, dtype=self.hparams['dtype'])
            self.f = torch.tensor(self.f_d, dtype=self.hparams['dtype'])
            self.etab = torch.tensor(self.eta_y0_d, dtype=self.hparams['dtype'])
            self.etat = torch.tensor(self.eta_yL_d, dtype=self.hparams['dtype'])
            self.gl = torch.tensor(self.g_x0_d, dtype=self.hparams['dtype'])
            self.gr = torch.tensor(self.g_xL_d, dtype=self.hparams['dtype']) 
            self.u0 = torch.tensor(self.u0_d, dtype=self.hparams['dtype'])             
            self.u = torch.tensor(self.u_d, dtype=self.hparams['dtype'])             
            dataset = torch.utils.data.TensorDataset(self.theta, self.f, self.etab, self.etat, self.gl, self.gr, self.u0, self.u)
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO' or self.hparams['modeltype']=='matrix data NGO':
            self.F = torch.tensor(self.F, dtype=self.hparams['dtype'])
            self.d = torch.tensor(self.d, dtype=self.hparams['dtype'])
            self.C = torch.tensor(self.C, dtype=self.hparams['dtype'])
            self.C_m = torch.tensor(self.C_m, dtype=self.hparams['dtype'])
            self.u = torch.tensor(self.u_d, dtype=self.hparams['dtype'])   
            dataset = torch.utils.data.TensorDataset(self.F, self.d, self.C, self.C_m, self.u)
        self.trainingset, self.validationset = random_split(dataset, [self.hparams['N_samples_train'], self.hparams['N_samples_val']])

    def train_dataloader(self):
        return DataLoader(self.trainingset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=0, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.validationset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=0, pin_memory=False)