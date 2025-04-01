# Copyright 2025 Joost Prins

# Standard
# import json

# 3rd Party
# import opt_einsum
import numpy as np
import torch
import torch.utils.data as torch_data
import pytorch_lightning as pl

# Local
from ngo.testproblems.tdarcy.NeuralOperator import NeuralOperator
from ngo.testproblems.tdarcy.manufacturedsolutions import ManufacturedSolutionsSet
# import ngo.ml.quadrature as quadrature
# from quadrature import *
# from basisfunctions import *

class DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.N_samples = self.hparams['N_samples_train'] + self.hparams['N_samples_val']
        dummymodel = NeuralOperator(self.hparams)
        # Generate input and output functions
        print('Generating functions...')
        dataset = ManufacturedSolutionsSet(N_samples=self.N_samples, variables=self.hparams['variables'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'])
        theta = dataset.theta
        f = dataset.f
        eta_y0 = dataset.etab
        eta_yL = dataset.etat
        g_x0 = dataset.gl
        g_xL = dataset.gr
        u = dataset.u
        u0 = dataset.u0
        #Discretize input functions
        # if self.hparams['modeltype']=='NN' or self.hparams['modeltype']=='DeepONet' or self.hparams['modeltype']=='VarMiON':
        print('Discretizing input functions...')
        self.theta_q, self.theta_x0_q, self.theta_xL_q, self.f_q, self.eta_y0_q, self.eta_yL_q, self.g_x0_q, self.g_xL_q, self.u0_t0_q = dummymodel.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
        if dummymodel.hparams['modeltype']=='model NGO':
            print('Assembling F...')
            bs = self.hparams['assembly_batch_size']
            n_batches = int(self.N_samples/bs)
            self.F = np.zeros((len(theta),self.hparams['N'],self.hparams['N']))
            for i in range(n_batches):
                print('batch: '+str(i))
                self.F[bs*i:bs*(i+1)] = dummymodel.compute_F(self.theta_q[bs*i:bs*(i+1)], self.theta_x0_q[bs*i:bs*(i+1)], self.theta_xL_q[bs*i:bs*(i+1)])
        if dummymodel.hparams['modeltype']=='data NGO':
            print('Assembling F...')
            self.F = dummymodel.compute_F(self.theta_q, self.theta_x0_q, self.theta_xL_q)
        if dummymodel.hparams['modeltype']=='model NGO' or dummymodel.hparams['modeltype']=='data NGO':
            print('Assembling d...')
            self.d = dummymodel.compute_d(self.f_q, self.eta_y0_q, self.eta_yL_q, self.g_x0_q, self.g_xL_q, self.u0_t0_q)
            print('Calculating conserved quantity...')
            self.C = dummymodel.compute_C(self.f_q, self.eta_y0_q, self.eta_yL_q, self.u0_t0_q)
            self.C_m = dummymodel.compute_C_m(self.theta_x0_q, self.theta_xL_q)
        print('Discretizing output function...')
        self.u = dummymodel.discretize_output_function(u)    
        if dummymodel.hparams['output_coefficients']==True:
            self.u = dummymodel.project_output_function(self.u)
        
    def setup(self, stage=None):
        torch.cuda.empty_cache()
        if self.hparams['modeltype']=='NN' or self.hparams['modeltype']=='DeepONet' or self.hparams['modeltype']=='VarMiON':
            self.theta_q = torch.tensor(self.theta_q, dtype=self.hparams['dtype'])
            self.f_q = torch.tensor(self.f_q, dtype=self.hparams['dtype'])
            self.etab_q = torch.tensor(self.eta_y0_q, dtype=self.hparams['dtype'])
            self.etat_q = torch.tensor(self.eta_yL_q, dtype=self.hparams['dtype'])
            self.gl_q = torch.tensor(self.g_x0_q, dtype=self.hparams['dtype'])
            self.gr_q = torch.tensor(self.g_xL_q, dtype=self.hparams['dtype']) 
            self.u0_t0_q = torch.tensor(self.u0_t0_q, dtype=self.hparams['dtype'])             
            self.u = torch.tensor(self.u, dtype=self.hparams['dtype'])             
            dataset = torch.utils.data.TensorDataset(self.theta_q, self.f_q, self.etab_q, self.etat_q, self.gl_q, self.gr_q, self.u0_t0_q, self.u)
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO' or self.hparams['modeltype']=='matrix data NGO':
            self.F = torch.tensor(self.F, dtype=self.hparams['dtype'])
            self.d = torch.tensor(self.d, dtype=self.hparams['dtype'])
            self.C = torch.tensor(self.C, dtype=self.hparams['dtype'])
            self.C_m = torch.tensor(self.C_m, dtype=self.hparams['dtype'])
            self.u = torch.tensor(self.u, dtype=self.hparams['dtype'])   
            dataset = torch.utils.data.TensorDataset(self.F, self.d, self.C, self.C_m, self.u)
        self.trainingset, self.validationset = torch_data.random_split(dataset, [self.hparams['N_samples_train'], self.hparams['N_samples_val']])

    def train_dataloader(self):
        return torch_data.DataLoader(self.trainingset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=0, pin_memory=False)

    def val_dataloader(self):
        return torch_data.DataLoader(self.validationset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=0, pin_memory=False)