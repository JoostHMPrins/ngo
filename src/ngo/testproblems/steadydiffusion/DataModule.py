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
    """
    LightningDataModule for handling data loading and preprocessing for the neural operator model.

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
        - Initialize the DataModule with hyperparameters.
        - Define N_samples (int): Total number of samples (training + validation)
        - Define a dummy model, used to discretize and preprocess training data
        - Generate a manufactured solutions function set
        - Discretize the functions onto the quadrature grid in the interior and on the boundaries
        - Assemble the system matrix/vector F
        - Define the scaling factor in case scale equivariance is enabled
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