# Copyright 2025 Joost Prins

# 3rd Party
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

# Local
from ngo.testproblems.steadydiffusion.NeuralOperator import NeuralOperator
from ngo.testproblems.steadydiffusion.manufacturedsolutions import ManufacturedSolutionsSet
# from ngo.ml.quadrature import UniformQuadrature, GaussLegendreQuadrature


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
        eta_y0 = dataset.eta_y0
        eta_yL = dataset.eta_yL
        g_x0 = dataset.g_x0
        g_xL = dataset.g_xL
        u = dataset.u
        #Discretize input functions
        print('Discretizing functions...')
        self.theta_q, self.theta_x0_q, self.theta_xL_q, self.f_q, self.eta_y0_q, self.eta_yL_q, self.g_x0_q, self.g_xL_q = dummymodel.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        self.u_q = dummymodel.discretize_output_function(u)
        print('Assembling system...')
        if dummymodel.hparams['modeltype']=='model NGO' or dummymodel.hparams['modeltype']=='data NGO':
                self.F = dummymodel.compute_F(self.theta_q, self.theta_x0_q, self.theta_xL_q)
                self.d = dummymodel.compute_d(self.f_q, self.eta_y0_q, self.eta_yL_q, self.g_x0_q, self.g_xL_q)
        self.scaling = np.abs(np.sum(dummymodel.w_Omega[None,:]*self.theta_q, axis=-1))

    def setup(self, stage=None):
        if self.hparams['modeltype']=='NN' or self.hparams['modeltype']=='DeepONet' or self.hparams['modeltype']=='VarMiON':
            self.theta_q = torch.tensor(self.theta_q, dtype=self.hparams['dtype'])
            self.f_q = torch.tensor(self.f_q, dtype=self.hparams['dtype'])
            self.eta_y0_q = torch.tensor(self.eta_y0_q, dtype=self.hparams['dtype'])
            self.eta_yL_q = torch.tensor(self.eta_yL_q, dtype=self.hparams['dtype'])
            self.g_x0_q = torch.tensor(self.g_x0_q, dtype=self.hparams['dtype'])
            self.g_xL_q = torch.tensor(self.g_xL_q, dtype=self.hparams['dtype']) 
            self.u_q = torch.tensor(self.u, dtype=self.hparams['dtype'])             
            dataset = torch.utils.data.TensorDataset(self.theta_q, self.f_q, self.eta_y0_q, self.eta_yL_q, self.g_x0_q, self.g_xL_q, self.u_q)
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO' or self.hparams['modeltype']=='matrix data NGO':
            self.scaling = torch.tensor(self.scaling, dtype=self.hparams['dtype'])            
            self.F = torch.tensor(self.F, dtype=self.hparams['dtype'])
            self.d = torch.tensor(self.d, dtype=self.hparams['dtype'])
            self.u_q = torch.tensor(self.u_q, dtype=self.hparams['dtype'])   
            dataset = torch.utils.data.TensorDataset(self.scaling, self.F, self.d, self.u_q)
        self.trainingset, self.validationset = random_split(dataset, [self.hparams['N_samples_train'], self.hparams['N_samples_val']])

    def train_dataloader(self):
        return DataLoader(self.trainingset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=0, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.validationset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=0, pin_memory=False)