# Copyright 2025 Joost Prins

# 3rd Party
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

# Local
from ngo.testproblems.steadydiffusion.NeuralOperator import NeuralOperator
from ngo.trainingdata.GRF import ManufacturedSolution
from ngo.testproblems.steadydiffusion.manufacturedsolutions import *


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
        u_q (np.ndarray): Discretized output function u, shape (N_samples, ...).
        F (np.ndarray): Computed function F, shape (N_samples, ...).
        d (np.ndarray): Computed function d, shape (N_samples, ...).
        scaling (np.ndarray): Scaling factor, shape (N_samples, ...).
        trainingset (torch.utils.data.Dataset): Training dataset.
        validationset (torch.utils.data.Dataset): Validation dataset.
    """

    def __init__(self, hparams):
        """
        Initialize the DataModule with hyperparameters and generate the dataset.
        Args:
            hparams (dict): Hyperparameter dictionary.
        """
        super().__init__()
        # Initialize hyperparameters
        self.hparams.update(hparams)
        # Define N_samples (int): Total number of samples (training + validation)
        self.N_samples = self.hparams['N_samples_train'] + self.hparams['N_samples_val']
        # Define a dummy model, used to discretize and preprocess training data
        dummymodel = NeuralOperator(self.hparams)
        #Generate manufactured solutions
        u = ManufacturedSolution(self.N_samples, # Number of samples
                                 self.hparams['d'], # Number of spatial dimensions
                                 self.hparams['l_min'], # Minimum GRF length scale per dimension
                                 self.hparams['l_max'], # Maximum GRF length scale per dimension
                                 self.hparams['b_u_min'], # minimum solution field mean
                                 self.hparams['b_u_max'], # maximum solution field mean
                                 self.hparams['c_u_min'], # minimum solution field standard deviation
                                 self.hparams['c_u_max'], # maximum solution field standard deviation
                                 self.hparams['device'], # Device to perform computations on
                                 self.hparams['dtype'], # Data type for computations
                                 self.hparams['n_RBF']) # Number of basis functions per dimension for data generation
        theta = ManufacturedSolution(self.N_samples, # Number of samples
                            self.hparams['d'], # Number of spatial dimensions
                            self.hparams['l_min'], # Minimum GRF length scale per dimension
                            self.hparams['l_max'], # Maximum GRF length scale per dimension
                            self.hparams['b_theta_min'], # minimum coefficient field mean
                            self.hparams['b_theta_max'], # maximum coefficient field mean
                            self.hparams['c_theta_min'], # minimum coefficient field standard deviation
                            self.hparams['c_theta_max'], # maximum coefficient field standard deviation
                            self.hparams['device'], # Device to perform computations on
                            self.hparams['dtype'], # Data type for computations
                            self.hparams['n_RBF']) # Number of basis functions per dimension for data generation
        f = Forcing(theta, u)
        n_y0 = torch.tensor([0,-1], dtype=self.hparams['dtype'], device=self.hparams['device']) #Normal vector at y0 boundary
        n_yL = torch.tensor([0,1], dtype=self.hparams['dtype'], device=self.hparams['device']) #Normal vector at yL boundary
        eta_y0, eta_yL = NeumannBC(n_y0, theta, u), NeumannBC(n_yL, theta, u)
        g_x0, g_xL = DirichletBC(theta, u), DirichletBC(theta, u)
        #Discretize input functions
        self.theta_q, self.theta_x0_q, self.theta_xL_q, self.f_q, self.eta_y0_q, self.eta_yL_q, self.g_x0_q, self.g_xL_q = dummymodel.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        self.u_q = dummymodel.discretize_output_function(u)
        print('Assembling system...')
        if dummymodel.hparams['modeltype']=='model NGO' or dummymodel.hparams['modeltype']=='data NGO':
                self.F = dummymodel.compute_F(self.theta_q, self.theta_x0_q, self.theta_xL_q)
                self.d = dummymodel.compute_d(self.f_q, self.eta_y0_q, self.eta_yL_q, self.g_x0_q, self.g_xL_q)
        self.scaling = np.abs(np.sum(dummymodel.w_Omega[None,:]*self.theta_q, axis=-1))

    def setup(self, stage=None):
        """
        Set up the dataset for training and validation. Setup is different according to the choice of modeltype.

        Args:
        stage (str, optional): Stage of the setup process. Defaults to None.

        Returns:
        None
        """
        if self.hparams['modeltype']=='NN' or self.hparams['modeltype']=='DeepONet' or self.hparams['modeltype']=='VarMiON':
            self.theta_q = torch.tensor(self.theta_q, dtype=self.hparams['dtype'])
            self.f_q = torch.tensor(self.f_q, dtype=self.hparams['dtype'])
            self.eta_y0_q = torch.tensor(self.eta_y0_q, dtype=self.hparams['dtype'])
            self.eta_yL_q = torch.tensor(self.eta_yL_q, dtype=self.hparams['dtype'])
            self.g_x0_q = torch.tensor(self.g_x0_q, dtype=self.hparams['dtype'])
            self.g_xL_q = torch.tensor(self.g_xL_q, dtype=self.hparams['dtype']) 
            self.u_q = torch.tensor(self.u_q, dtype=self.hparams['dtype'])             
            dataset = torch.utils.data.TensorDataset(self.theta_q, self.f_q, self.eta_y0_q, self.eta_yL_q, self.g_x0_q, self.g_xL_q, self.u_q)
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO' or self.hparams['modeltype']=='matrix data NGO':
            self.scaling = torch.tensor(self.scaling, dtype=self.hparams['dtype'])            
            self.F = torch.tensor(self.F, dtype=self.hparams['dtype'])
            self.d = torch.tensor(self.d, dtype=self.hparams['dtype'])
            self.u_q = torch.tensor(self.u_q, dtype=self.hparams['dtype'])   
            dataset = torch.utils.data.TensorDataset(self.scaling, self.F, self.d, self.u_q)
        self.trainingset, self.validationset = random_split(dataset, [self.hparams['N_samples_train'], self.hparams['N_samples_val']])

    def train_dataloader(self):
        """
        Create the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """

        return DataLoader(self.trainingset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=0, pin_memory=False)

    def val_dataloader(self):
        """
        Create the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(self.validationset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=0, pin_memory=False)
