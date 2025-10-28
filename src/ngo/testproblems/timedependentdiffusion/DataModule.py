
# Copyright 2025 Joost Prins

# 3rd Party
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import opt_einsum

# Local
from ngo.testproblems.timedependentdiffusion.NeuralOperator import NeuralOperator
from ngo.testproblems.timedependentdiffusion.manufacturedsolutions import ManufacturedSolutionsSet

class DataModule(pl.LightningDataModule):
    """
    LightningDataModule for handling data generation and preprocessing for the neural operator model.

    Attributes:
        hparams (dict): Hyperparameter dictionary.
        N_samples (int): Total number of samples (training + validation).
        theta_q (np.ndarray): Discretized input function theta, shape (N_samples, ...).
        theta_x0_q (np.ndarray): Discretized input function theta at x0, shape (N_samples, ...).
        theta_xL_q (np.ndarray): Discretized input function theta at xL, shape (N_samples, ...).
        f_q (np.ndarray): Discretized input function f, shape (N_samples, ...).
        eta_y0_q (np.ndarray): Discretized input function eta at y0, shape (N_samples, ...).
        eta_yL_q (np.ndarray): Discretized input function eta at yL, shape (N_samples, ...).
        g_x0_q (np.ndarray): Discretized input function g at x0, shape (N_samples, ...).
        g_xL_q (np.ndarray): Discretized input function g at xL, shape (N_samples, ...).
        u0_t0_q (np.ndarray): Discretized initial condition u0 at t0, shape (N_samples, ...).
        u_q (np.ndarray): Discretized output function u, shape (N_samples, ...).
        F (np.ndarray): Computed function F, shape (N_samples, ...).
        d (np.ndarray): Computed function d, shape (N_samples, ...).
        d_x (np.ndarray): Computed spatial derivative of d, shape (N_samples, ...).
        C (np.ndarray): Computed conserved quantity C, shape (N_samples, ...).
        C_m (np.ndarray): Computed conserved quantity per dof C_m, shape (N_samples, ...).
        theta_bar (np.ndarray): Average of theta over the domain, shape (N_samples, ...).
        u_0_m (np.ndarray): Projection of initial condition onto basis functions, shape (N_samples, ...).
        trainingset (torch.utils.data.Dataset): Training dataset.
        validationset (torch.utils.data.Dataset): Validation dataset.
    """
    def __init__(self, hparams):
        """
        Initialize the DataModule.

        Args:
            hparams (dict): Hyperparameter dictionary.
        """
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
        u0 = dataset.u0
        #Discretize input functions
        print('Discretizing input functions...')
        self.theta_q, self.theta_x0_q, self.theta_xL_q, self.f_q, self.eta_y0_q, self.eta_yL_q, self.g_x0_q, self.g_xL_q, self.u0_t0_q = dummymodel.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
        if dummymodel.hparams['modeltype']=='model NGO' or dummymodel.hparams['modeltype']=='data NGO':
            print('Assembling F...')
            self.F = dummymodel.compute_F(self.theta_q, self.theta_x0_q, self.theta_xL_q)/self.hparams['Dt']
            print('Assembling d...')
            self.d = dummymodel.compute_d(self.f_q, self.eta_y0_q, self.eta_yL_q, self.g_x0_q, self.g_xL_q, self.u0_t0_q)/self.hparams['Dt']
            self.d_x = dummymodel.compute_d_x(self.f_q, self.eta_y0_q, self.eta_yL_q, self.g_x0_q, self.g_xL_q, self.u0_t0_q)/self.hparams['Dt']
            print('Calculating conserved quantity...')
            self.C = dummymodel.compute_C(self.f_q, self.eta_y0_q, self.eta_yL_q, self.g_x0_q, self.g_xL_q, self.u0_t0_q)/self.hparams['Dt']
            self.C_m = dummymodel.compute_C_m(self.theta_x0_q, self.theta_xL_q)/self.hparams['Dt']
            self.theta_bar = opt_einsum.contract('q,Nq->N', dummymodel.w_OmegaT, self.theta_q)/self.hparams['Dt']
            self.u_0_m = dummymodel.project_on_psi_tT(self.u0_t0_q)
        print('Discretizing output function...')
        self.u = dummymodel.discretize_output_function(u)    
        if dummymodel.hparams['output_coefficients']==True:
            self.u = dummymodel.project_output_function(self.u)

    def setup(self, stage=None):
        """
        Set up the dataset for training and validation. Setup is different according to the choice of modeltype.

        Args:
        stage (str, optional): Stage of the setup process. Defaults to None.

        Returns:
        None
        """
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
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO':
            self.F = torch.tensor(self.F, dtype=self.hparams['dtype'])
            self.d = torch.tensor(self.d, dtype=self.hparams['dtype'])
            self.d_x = torch.tensor(self.d_x, dtype=self.hparams['dtype'])
            self.u_0_m = torch.tensor(self.u_0_m, dtype=self.hparams['dtype'])
            self.C = torch.tensor(self.C, dtype=self.hparams['dtype'])
            self.C_m = torch.tensor(self.C_m, dtype=self.hparams['dtype'])
            self.theta_bar = torch.tensor(self.theta_bar, dtype=self.hparams['dtype'])      
            self.u = torch.tensor(self.u, dtype=self.hparams['dtype'])   
            dataset = torch.utils.data.TensorDataset(self.F, self.d, self.d_x, self.u_0_m, self.C, self.C_m, self.theta_bar, self.u)
        self.trainingset, self.validationset = random_split(dataset, [self.hparams['N_samples_train'], self.hparams['N_samples_val']])

    def train_dataloader(self):
        """
        Create the training dataloader.

        Returns:
            DataLoader: Training dataloader.
        """
        return DataLoader(self.trainingset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=0, pin_memory=False)

    def val_dataloader(self):
        """
        Create the validation dataloader.

        Returns:
            DataLoader: Validation dataloader.
        """
        return DataLoader(self.validationset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=0, pin_memory=False)