# Copyright 2025 Joost Prins

# 3rd Party
import torch
import numpy as np
import pytorch_lightning as pl
import opt_einsum

# Local
from ngo.numericalanalysis.basisfunctions import TensorProductBasis
from ngo.numericalanalysis.quadrature import GaussLegendreQuadrature, UniformQuadrature
from ngo.numericalanalysis.linalg import *
from ngo.ml.systemnets import MLP, LBranchNet
from ngo.ml.customlayers import Neumann_preconditioner_forward
from ngo.trainingdata.GRF import UnitFunction

class NeuralOperator(pl.LightningModule):
    """
    Neural Operator class for solving a steady diffusion equation.
    Mainly for the use of Neural Green's Operators (NGOs), and to benchmark these against other models from literature.
    modeltype options are: 'NN', 'DeepONet', 'VarMiON', 'data NGO', 'model NGO', 'FEM' and 'projection'.

    Attributes:
        hparams (dict): Hyperparameters for the model.
        systemnet (torch.nn.Module): The system network.
        basis_test (TensorProductBasis): Basis functions for testing.
        basis_trial (TensorProductBasis): Basis functions for trial.
        Identity (torch.Tensor): Identity matrix, shape (N, N).
        M_i (list of torch.Tensor): List of mass matrices per dimension.
    """
    def __init__(self, hparams):
        """
        Initialize the NeuralOperator class.

        Args:
        hparams (dict): Hyperparameters for the model.
        """
        super().__init__()
        # Initialize hyperparameters
        self.hparams.update(hparams)
        # Set device and data type
        self.to(self.hparams['device'])
        self.to(self.hparams['dtype'])
        # Compute quadrature points and weights
        self.compute_quadrature()
        # Initialize test and trial basis functions
        self.psi = TensorProductBasis(self.hparams['test_bases']) 
        self.phi = TensorProductBasis(self.hparams['trial_bases'])
        # Initialize model type and set forward functions
        self.init_modeltype()
        # Count trainable parameters
        self.hparams['N_w_real'] = sum(p.numel() for p in self.parameters())
        # Initialize system net
        self.systemnet = self.hparams['systemnet'](self.hparams).to(self.device)
        # Identity
        self.Identity = torch.eye(self.hparams['N'], dtype=self.dtype, device=self.device)
        # Assemble mass matrices
        self.M_i = self.assemble_M_i()

    def init_modeltype(self):
        """
        Initialize the model type and set the forward functions and corresponding input and output shapes.

        Returns:
            None
        """
        if self.hparams['modeltype']=='NN':
            self.forwardfunction = self.NN_forward
            self.simforwardfunction = self.NN_simforward
            self.hparams['input_shape'] = (4,)+(self.hparams['Q'])
            self.hparams['output_shape'] = self.hparams['Q_L']
        if self.hparams['modeltype']=='DeepONet':
            self.forwardfunction = self.DeepONet_forward
            self.simforwardfunction = self.DeepONet_simforward
            self.hparams['input_shape'] = (4,)+(self.hparams['Q'])
            self.hparams['output_shape'] = self.hparams['N_i']
        if self.hparams['modeltype']=='VarMiON':
            self.LBranch_f = LBranchNet(self.hparams, input_dim=self.hparams['Q'][0]*self.hparams['Q'][1], output_dim=self.hparams['N']).to(self.device)
            self.LBranch_eta = LBranchNet(self.hparams, input_dim=2*self.hparams['Q'][0], output_dim=self.hparams['N']).to(self.device)
            self.LBranch_g = LBranchNet(self.hparams, input_dim=2*self.hparams['Q'][1], output_dim=self.hparams['N']).to(self.device)
            self.forwardfunction = self.VarMiON_forward
            self.simforwardfunction = self.VarMiON_simforward
            self.hparams['input_shape'] = (1,)+(self.hparams['Q'])
            self.hparams['output_shape'] = (self.hparams['N'],self.hparams['N'])
        if self.hparams['modeltype']=='data NGO':
            self.forwardfunction = self.NGO_forward
            self.simforwardfunction = self.NGO_simforward
            self.hparams['input_shape'] = (1,)+(self.hparams['N_i'])
            self.hparams['output_shape'] = (self.hparams['N'],self.hparams['N'])
        if self.hparams['modeltype']=='model NGO':
            self.forwardfunction = self.NGO_forward
            self.simforwardfunction = self.NGO_simforward
            self.hparams['input_shape'] = (1,self.hparams['N'],self.hparams['N'])
            self.hparams['output_shape'] = (self.hparams['N'],self.hparams['N'])
            #Assemble Neumann preconditioner if needed
            if self.hparams['Neumannseries']==True:
                self.assemble_Neumann_preconditioner()   
        if self.hparams['modeltype']=='FEM':
            self.forwardfunction = self.FEM_forward
            self.simforwardfunction = self.FEM_simforward
            #Assemble Neumann preconditioner if needed
            if self.hparams['Neumannseries']==True:
                self.assemble_Neumann_preconditioner()   
        if self.hparams['modeltype']=='projection':
            self.forwardfunction = self.projection_forward
            self.simforwardfunction = self.projection_simforward
            #Assemble Neumann preconditioner if needed
            if self.hparams['Neumannseries']==True:
                self.assemble_Neumann_preconditioner()   

    def discretize_input_functions(self, theta, f, eta_y0, eta_yL, g_x0, g_xL):
        """
        Discretize the input functions.

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)

        Returns:
            tuple: Discretized input functions (theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q),
            each of shape (B, Q_0, ..., Q_{d-1}) (depending on quadrature shape).
        """
        theta_q = theta.forward(self.Xi_Omega)
        f_q = f.forward(self.Xi_Omega)
        eta_y0_q = eta_y0.forward(self.Xi_Gamma_y0)
        eta_yL_q = eta_yL.forward(self.Xi_Gamma_yL)
        g_x0_q = g_x0.forward(self.Xi_Gamma_x0)
        g_xL_q = g_xL.forward(self.Xi_Gamma_xL)
        return theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q
    
    def discretize_output_function(self, u):
        """
        Discretize the output function.

        Args:
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Discretized output function u_q, shape (B, Q_0, ..., Q_{d-1}).
        """
        u_q = u.forward(self.Xi_Omega_L)
        return u_q
    
    def assemble_F(self, theta):
        """
        Assemble the operator F based on the model type.

        Args:
            theta (ManufacturedSolution)

        Returns:
            torch.Tensor: Assembled operator F.
        """
        if  self.hparams['modeltype']=='data NGO':
            F = assemble_L(self.psi, theta, self.W_Omega, self.Xi_Omega, derivative_orders=(0,0))
        if  self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM' or self.hparams['modeltype']=='projection':
            # Volume term - derivative in x direction
            T_i = assemble_T(theta, self.psi, self.phi, self.W_Omega, self.Xi_Omega, derivative_orders=((0,0),(1,0),(1,0)))
            F = kron_vecT(theta.u_m, T_i)
            # Volume term - derivative in y direction
            T_i = assemble_T(theta, self.psi, self.phi, self.W_Omega, self.Xi_Omega, derivative_orders=((0,0),(0,1),(0,1)))
            F += kron_vecT(theta.u_m, T_i)
            # Boundary term at x0 (coming from Neumann complement)
            T_i = assemble_T(theta, self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0),(0,0),(1,0)))
            F += kron_vecT(theta.u_m, T_i)
            # Boundary term at xL (coming from Neumann complement)
            T_i = assemble_T(theta, self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0),(0,0),(1,0)))
            F -= kron_vecT(theta.u_m, T_i)
            # Boundary term at x0 (coming from Dirichlet complement)
            T_i = assemble_T(theta, self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0),(1,0),(0,0)))
            F += kron_vecT(theta.u_m, T_i)
            # Boundary term at xL (coming from Dirichlet complement)
            T_i = assemble_T(theta, self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0),(1,0),(0,0)))
            F -= kron_vecT(theta.u_m, T_i)
            if self.hparams.get('gamma_stabilization',0)!=0:
                # Stabilization term at x0
                T_i = assemble_T(theta, self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0),(0,0),(0,0)))
                F += self.hparams['gamma_stabilization']*kron_vecT(theta.u_m, T_i)
                # Stabilization term at xL
                T_i = assemble_T(theta, self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0),(0,0),(0,0)))
                F += self.hparams['gamma_stabilization']*kron_vecT(theta.u_m, T_i)
        return F
    
    def assemble_d(self, f, eta_y0, eta_yL, g_x0, g_xL):
        """
        Compute the right-hand side vector d.

        Args:
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)

        Returns:
            torch.Tensor: Right-hand side vector d, shape (B, N).
        """
        d = assemble_L(self.psi, f, self.W_Omega, self.Xi_Omega, derivative_orders=(0,0))
        d += assemble_L(self.psi, eta_y0, self.W_Gamma_y0, self.Xi_Gamma_y0, derivative_orders=(0,0))
        d += assemble_L(self.psi, eta_yL, self.W_Gamma_yL, self.Xi_Gamma_yL, derivative_orders=(0,0))
        d += assemble_L(self.psi, g_x0, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=(1,0)) # addition because of minus sign of boundary normal sign at x0
        d -= assemble_L(self.psi, g_xL, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=(1,0))
        if self.hparams.get('gamma_stabilization',0)!=0:
            d += self.hparams['gamma_stabilization']*assemble_L(self.psi, g_x0, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=(0,0))
            d += self.hparams['gamma_stabilization']*assemble_L(self.psi, g_xL, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=(0,0))
        d = d.flatten(start_dim=1)
        return d
    
    def assemble_M_i(self):
        """
        Compute the mass matrix M_i.

        Returns:
            M_i (tuple of torch.Tensor): Mass matrix M_i per dimension.
        """ 
        M_i = assemble_B(self.phi, self.phi, self.W_Omega, self.Xi_Omega, derivative_orders=((0,0),(0,0)))
        return M_i
    
    # def assemble_Neumann_preconditioner(self):
    #     """
    #     Compute the matrices F_0 and A_0 for the Neumann series preconditioner.

    #     Returns:
    #         None
    #     """ 
    #     # Volume term - derivative in x direction
    #     F_0 = kron(assemble_B(self.psi, self.phi, self.W_Omega, self.Xi_Omega, derivative_orders=((1,0),(1,0))))
    #     # Volume term - derivative in y direction
    #     F_0 += kron(assemble_B(self.psi, self.phi, self.W_Omega, self.Xi_Omega, derivative_orders=((0,1),(0,1))))
    #     # Boundary term at x0 (coming from Neumann complement)
    #     F_0 += kron(assemble_B(self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0),(1,0))))
    #     # Boundary term at xL (coming from Neumann complement)
    #     F_0 -= kron(assemble_B(self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0),(1,0))))
    #     # Boundary term at x0 (coming from Dirichlet complement)
    #     F_0 += kron(assemble_B(self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((1,0),(0,0))))
    #     # Boundary term at xL (coming from Dirichlet complement)
    #     F_0 -= kron(assemble_B(self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((1,0),(0,0))))
    #     if self.hparams.get('gamma_stabilization',0)!=0:
    #         # Stabilization term at x0
    #         F_0 += self.hparams['gamma_stabilization']*kron(assemble_B(self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0),(0,0))))
    #         # Stabilization term at xL
    #         F_0 += self.hparams['gamma_stabilization']*kron(assemble_B(self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0),(0,0))))
    #     self.F_0 = F_0
    #     self.A_0 = torch.linalg.inv(F_0)

    def assemble_Neumann_preconditioner(self):
        """
        Compute the matrices F_0 and A_0 for the Neumann series preconditioner.

        Returns:
            None
        """ 
        theta = UnitFunction(d=self.hparams['d'], device=self.device, dtype=self.dtype)
        self.F_0 = self.assemble_F(theta)
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM' or self.hparams['modeltype']=='projection':
            self.A_0 = torch.linalg.inv(self.F_0)
    
    def NN_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q):
        """
        Train forward pass for the NN model (input functions -> solution).

        Args:
            theta_q (torch.Tensor): Discretized input function theta, shape (B, Q_0, Q_1).
            f_q (torch.Tensor): Discretized input function f, shape (B, Q_0, Q_1).
            eta_y0_q (torch.Tensor): Discretized input function eta_y0, shape (B, Q_0, 1).
            eta_yL_q (torch.Tensor): Discretized input function eta_yL, shape (B, Q_0, 1).
            g_x0_q (torch.Tensor): Discretized input function g_x0, shape (B, 1, Q_1).
            g_xL_q (torch.Tensor): Discretized input function g_xL, shape (B, 1, Q_1).

        Returns:
            torch.Tensor: Predicted solution u_q_hat, shape (B, Q_0, Q_1, Q_2).
        """    
        theta_q = theta_q.reshape((theta_q.shape[0],)+self.hparams['Q'])
        f_q = f_q.reshape((f_q.shape[0],)+self.hparams['Q'])
        eta_q = torch.zeros((eta_y0_q.shape[0],)+self.hparams['Q'], dtype=self.dtype, device=self.device)
        eta_q[:,:,0] = eta_y0_q[:,:,0]
        eta_q[:,:,-1] = eta_yL_q[:,:,0]
        g_q = torch.zeros((g_x0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        g_q[:,0,:] = g_x0_q[:,0,:]
        g_q[:,-1,:] = g_xL_q[:,0,:]
        inputfuncs = torch.stack((theta_q,f_q,eta_q,g_q), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta_q.flatten(-2,-1),f_q.flatten(-2,-1),eta_y0_q,eta_yL_q,g_x0_q,g_xL_q), dim=1)
        u_q_hat = self.systemnet.forward(inputfuncs)
        return u_q_hat
    
    def NN_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        """
        Simulation forward pass for the NN model (input functions -> solution).

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u (ManufacturedSolution)

        Returns:
            np.ndarray: Predicted solution u_q_hat, shape (B, Q_0, Q_1).
        """
        theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        theta_q = torch.tensor(theta_q, dtype=self.dtype, device=self.device)
        f_q = torch.tensor(f_q, dtype=self.dtype, device=self.device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.dtype, device=self.device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.dtype, device=self.device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.dtype, device=self.device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.dtype, device=self.device)   
        u_q_hat = self.NN_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q)
        return u_q_hat
    
    def DeepONet_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q):
        """
        Train forward pass for the DeepONet model.

        Args:
            theta_q (torch.Tensor): Discretized input function theta, shape (B, Q_0, Q_1).
            f_q (torch.Tensor): Discretized input function f, shape (B, Q_0, Q_1).
            eta_y0_q (torch.Tensor): Discretized input function eta_y0, shape (B, Q_0, 1).
            eta_yL_q (torch.Tensor): Discretized input function eta_yL, shape (B, Q_0, 1).
            g_x0_q (torch.Tensor): Discretized input function g_x0, shape (B, 1, Q_1).
            g_xL_q (torch.Tensor): Discretized input function g_xL, shape (B, 1, Q_1).

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """      
        theta_q = theta_q.reshape((theta_q.shape[0],)+self.hparams['Q'])
        f_q = f_q.reshape((f_q.shape[0],)+self.hparams['Q'])
        eta_q = torch.zeros((eta_y0_q.shape[0],)+self.hparams['Q'], dtype=self.dtype, device=self.device)
        eta_q[:,:,0] = eta_y0_q[:,:,0]
        eta_q[:,:,-1] = eta_yL_q[:,:,0]
        g_q = torch.zeros((g_x0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        g_q[:,0,:] = g_x0_q[:,0,:]
        g_q[:,-1,:] = g_xL_q[:,0,:]
        inputfuncs = torch.stack((theta_q,f_q,eta_q,g_q), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta_q.flatten(-2,-1),f_q.flatten(-2,-1),eta_y0_q,eta_yL_q,g_x0_q,g_xL_q), dim=1)
        u_m_hat = self.systemnet.forward(inputfuncs)
        return u_m_hat
    
    def DeepONet_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        """
        Simulation forward pass for the DeepONet model.

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u (ManufacturedSolution)            
        
        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
            """
        theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        theta_q = torch.tensor(theta_q, dtype=self.dtype, device=self.device)
        f_q = torch.tensor(f_q, dtype=self.dtype, device=self.device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.dtype, device=self.device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.dtype, device=self.device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.dtype, device=self.device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.dtype, device=self.device)   
        u_m_hat = self.DeepONet_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q)
        return u_m_hat
    
    def VarMiON_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q):
        """
        Train forward pass for the VarMiON model.

        Args:
            theta_q (torch.Tensor): Discretized input function theta, shape (B, Q_0, Q_1).
            f_q (torch.Tensor): Discretized input function f, shape (B, Q_0, Q_1).
            eta_y0_q (torch.Tensor): Discretized input function eta_y0, shape (B, Q_0, 1).
            eta_yL_q (torch.Tensor): Discretized input function eta_yL, shape (B, Q_0, 1).
            g_x0_q (torch.Tensor): Discretized input function g_x0, shape (B, 1, Q_1).
            g_xL_q (torch.Tensor): Discretized input function g_xL, shape (B, 1, Q_1).

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (N_samples, N).
        """
        f_q = f_q.flatten(start_dim=1)
        eta_y0_q = eta_y0_q.flatten(start_dim=1)
        eta_yL_q = eta_yL_q.flatten(start_dim=1)
        eta_q = torch.cat((eta_y0_q, eta_yL_q), dim=1)
        g_x0_q = g_x0_q.flatten(start_dim=1)
        g_xL_q = g_xL_q.flatten(start_dim=1)
        g_q = torch.cat((g_x0_q, g_xL_q), dim=1)
        systemnet = self.systemnet.forward(theta_q)
        LBranch = self.LBranch_f.forward(f_q) + self.LBranch_eta.forward(eta_q) + self.LBranch_g.forward(g_q)
        u_m = opt_einsum.contract('nij,nj->ni', systemnet, LBranch)
        return u_m
    
    def VarMiON_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        """
        Simulation forward pass for the VarMiON model.

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """ 
        theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        theta_q = torch.tensor(theta_q, dtype=self.dtype, device=self.device)
        f_q = torch.tensor(f_q, dtype=self.dtype, device=self.device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.dtype, device=self.device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.dtype, device=self.device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.dtype, device=self.device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.dtype, device=self.device)   
        u_m_hat = self.VarMiON_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q)
        return u_m_hat

    def NGO_forward(self, scaling, F, d):
        """
        Train forward pass for the NGO model.

        Args:
            scaling (torch.Tensor): Input tensor scaling, shape (B, N).
            F (torch.Tensor): Input tensor F, shape (B, N, N).
            d (torch.Tensor): Input tensor d, shape (B, N).

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """
        if self.hparams.get('scale_equivariance',False)==True:
            F = F/scaling[:,None,None]
        if self.hparams.get('Neumannseries', False)==False:
            A_hat = self.systemnet.forward(F)
        if self.hparams.get('Neumannseries', False)==True:
            A_hat = Neumann_preconditioner_forward(self, F)
        if self.hparams.get('scale_equivariance',False)==True:
            A_hat = A_hat/scaling[:,None,None]        
        u_m_hat = opt_einsum.contract('nij,nj->ni', A_hat, d)
        return u_m_hat
    
    def NGO_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        """
        Simulation forward pass for the NGO model.

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """
        F = self.assemble_F(theta)
        d = self.assemble_d(f, eta_y0, eta_yL, g_x0, g_xL)
        scaling = theta.basis.integrate(theta.u_m, self.W_Omega, self.Xi_Omega)
        u_m_hat = self.NGO_forward(scaling, F, d)
        return u_m_hat
    
    def FEM_forward(self, F, d):
        """
        Forward pass for the FEM model.

        Args:
            F (torch.Tensor): Input tensor F, shape (B, N, N).
            d (torch.Tensor): Input tensor d, shape (B, N).

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """
        if self.hparams.get('Neumannseries', False)==False:
            # F_inv = torch.linalg.inv(F)
            u_m_hat = torch.linalg.solve(F, d)
        if self.hparams['Neumannseries']==True:
            T = torch.zeros(F.shape, dtype=self.dtype, device=self.device)
            Ti = self.Identity
            T1 = -F@self.A_0 + self.Identity
            for i in range(0, self.hparams['Neumannseries_order']):
                Ti = T1@Ti
                T = T + Ti
            F_inv = self.A_0@(self.Identity + T)
            u_m_hat = opt_einsum.contract('nij,nj->ni', F_inv, d)
        return u_m_hat
    
    def FEM_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        """
        Simulation forward pass for the FEM model.

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """
        F = self.assemble_F(theta)
        d = self.assemble_d(f, eta_y0, eta_yL, g_x0, g_xL)
        u_m_hat = self.FEM_forward(F, d)
        return u_m_hat
    
    def projection_forward(self, u):
        """
        L2 projection of the solution onto the trial basis functions.

        Args:
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Projected solution coefficients u_m_hat, shape (N_samples, N).
        """
        d = assemble_L(self.phi, u, self.W_Omega, self.Xi_Omega, derivative_orders=(0,0))
        u_m_hat = kron_solve(self.M_i, d)      
        return u_m_hat
    
    def projection_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        """
        Simulation forward pass for the projection model.

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """
        u_m_hat = self.projection_forward(u)
        return u_m_hat
    
    def forward(self, *args):
        """
        Training forward pass for the model, set in the function self.init_modeltype(),
        based on the selected hparams['modeltype'].

        Args:
            *args: Variable length argument list.

        Returns:
            torch.Tensor: Predicted output u_q_hat, shape (B, Q_0, ..., Q_{d-1}).
            Or:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).    
        """
        if self.hparams['modeltype']=='NN':
            u_q_hat = self.forwardfunction(*args)
            return u_q_hat
        if self.hparams['modeltype']!='NN':
            u_m = self.forwardfunction(*args)
            if self.hparams['output_coefficients']==True:
                return u_m
            if self.hparams['output_coefficients']==False:
                u_q_hat = self.phi.forward(u_m, self.Xi_Omega_L)
                return u_q_hat

    def simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, X, u):
        """
        Simulation forward pass for the model, set in the function self.init_modeltype(),
        based on the selected hparams['modeltype'].
        
        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u0 (see manufacturedsolutions.py)
            X (torch.Tensor): Evaluation points, shape (Q_0, Q_1, ..., Q_{d-1}).
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Predicted output u_q_hat, shape (B, Q_0, Q_1, Q_2).
        """
        u_m_hat = self.simforwardfunction(theta, f, eta_y0, eta_yL, g_x0, g_xL, u)
        u_q_hat = u_m_hat if self.hparams['modeltype']=='NN' else self.phi.forward(u_m_hat, X)
        return u_q_hat

    def Neumann_criterion(self, theta):
        """
        Compute the spectral radius of the operator T1 = -F*A_0 to check the Neumann series convergence criterion.
        
        Args:
            theta (list of functions): Input function theta, shape len(N_samples).

        Returns:
            torch.Tensor: Spectral radius of T1, shape (N_samples,).
        """
        F = self.assemble_F(theta)
        if self.hparams.get('scaling_equivariance',False)==True:
            scaling = theta.basis.integrate(theta.u_m, self.W_Omega, self.Xi_Omega)
            F = F/scaling[:,None,None]
        T1 = -F@self.A_0 + self.Identity
        eigvals = torch.linalg.eigvals(T1)
        spectralradius = torch.amax(torch.real((eigvals*torch.conj(eigvals))**(1/2)), dim=-1)
        return spectralradius
    
    def compute_quadrature(self):
        """
        Compute the quadrature points and weights.

        Returns:
            None
        """
        # Define zero and one tensors
        zero = torch.zeros(size=(1,), dtype=self.dtype, device=self.device)
        one = torch.ones(size=(1,), dtype=self.dtype, device=self.device)
        if self.hparams['quadrature']=='uniform':
            self.w_x, self.xi_x = UniformQuadrature(Q=self.hparams['Q'][0], dtype=self.dtype, device=self.device).compute_quadrature()
            self.w_y, self.xi_y = UniformQuadrature(Q=self.hparams['Q'][1], dtype=self.dtype, device=self.device).compute_quadrature()
        if self.hparams['quadrature']=='Gauss-Legendre':
            self.w_x, self.xi_x = GaussLegendreQuadrature(Q=self.hparams['Q'][0], n_elements=self.hparams['n_elements'][0], dtype=self.dtype, device=self.device).compute_quadrature()
            self.w_y, self.xi_y = GaussLegendreQuadrature(Q=self.hparams['Q'][1], n_elements=self.hparams['n_elements'][1], dtype=self.dtype, device=self.device).compute_quadrature()
        # Domain quadrature
        self.W_Omega = self.w_x, self.w_y
        self.Xi_Omega = self.xi_x, self.xi_y
        # Boundary x0 quadrature
        self.W_Gamma_x0 = one, self.w_y
        self.Xi_Gamma_x0 = zero, self.xi_y
        # Boundary xL quadrature
        self.W_Gamma_xL = one, self.w_y
        self.Xi_Gamma_xL = one, self.xi_y
        # Boundary y0 quadrature
        self.W_Gamma_y0 = self.w_x, one
        self.Xi_Gamma_y0 = self.xi_x, zero
        # Boundary yL quadrature
        self.W_Gamma_yL = self.w_x, one
        self.Xi_Gamma_yL = self.xi_x, one
        #Loss quadrature
        if self.hparams['quadrature_L']=='uniform':
            self.w_x_L, self.xi_x_L = UniformQuadrature(Q=self.hparams['Q_L'][0], dtype=self.dtype, device=self.device).compute_quadrature()
            self.w_y_L, self.xi_y_L = UniformQuadrature(Q=self.hparams['Q_L'][1], dtype=self.dtype, device=self.device).compute_quadrature()
        if self.hparams['quadrature_L']=='Gauss-Legendre':
            self.w_x_L, self.xi_x_L = GaussLegendreQuadrature(Q=self.hparams['Q_L'][0], n_elements=self.hparams['n_elements_L'][0], dtype=self.dtype, device=self.device).compute_quadrature()
            self.w_y_L, self.xi_y_L = GaussLegendreQuadrature(Q=self.hparams['Q_L'][1], n_elements=self.hparams['n_elements_L'][1], dtype=self.dtype, device=self.device).compute_quadrature()
        # Loss domain quadrature
        self.W_Omega_L = self.w_x_L, self.w_y_L
        self.Xi_Omega_L = self.xi_x_L, self.xi_y_L

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        optimizer = self.hparams['optimizer'](self.parameters(), lr=self.hparams['learning_rate'])
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        """
        Definition of the training step (called under the hood by pytorch lightning).

        Args:
            train_batch (tuple): Batch of training data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        inputs = train_batch[:-1]
        loss = 0
        if self.hparams.get('solution_loss',None)!=None:
            u = train_batch[-1]
            u_hat = self.forward(*inputs)
            if self.hparams['output_coefficients']==True:
                loss += self.hparams['solution_loss'](self.M_i, u_hat, u)
            if self.hparams['output_coefficients']==False:
                loss += self.hparams['solution_loss'](self.W_Omega_L, u_hat, u)
        if self.hparams.get('matrix_loss',None)!=None:
            F = inputs[1]
            if self.hparams.get('Neumannseries', False)==False:
                A_hat = self.systemnet.forward(F)
            if self.hparams.get('Neumannseries', False)==True:
                A_hat = Neumann_preconditioner_forward(self, F)
            loss += self.hparams['matrix_loss'](torch.matmul(F, torch.matmul(A_hat,F)), F)
            self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Definition of the validation step (called under the hood by pytorch lightning).

        Args:
            val_batch (tuple): Batch of validation data.
            batch_idx (int): Index of the batch.

        Returns:
            None
        """
        inputs = val_batch[:-1]
        u = val_batch[-1]
        u_hat = self.forward(*inputs)
        loss = 0
        if self.hparams.get('solution_loss',None)!=None:
            if self.hparams['output_coefficients']==True:
                loss += self.hparams['solution_loss'](self.M_i, u_hat, u)
            if self.hparams['output_coefficients']==False:
                loss += self.hparams['solution_loss'](self.W_Omega_L, u_hat, u)
        if self.hparams.get('matrix_loss',None)!=None:
            F = inputs[1]
            if self.hparams.get('Neumannseries',False)==False:
                A_hat = self.systemnet.forward(F)
            if self.hparams.get('Neumannseries', False)==True:
                A_hat = Neumann_preconditioner_forward(self, F)
            loss += self.hparams['matrix_loss'](torch.matmul(F, torch.matmul(A_hat,F)), F)
            print(loss)
        if self.hparams['output_coefficients']==True:
            metric = self.hparams['metric'](self.M_i, u_hat, u)
        if self.hparams['output_coefficients']==False:
            metric = self.hparams['metric'](self.W_Omega_L, u_hat, u)
        self.metric = metric.item()
        self.log('val_loss', loss)
        self.log('metric', metric)

    def on_save_checkpoint(self, checkpoint):
        """
        Save hyperparameters to the checkpoint (hook called by pytorch lightning under the hood).

        Args:
            checkpoint (dict): Checkpoint dictionary.

        Returns:
            None
        """
        checkpoint['hparams'] = self.hparams
        
    def on_fit_start(self):
        """
        Perform actions at the start of fitting (hook called by pytorch lightning under the hood).

        Returns:
            None
        """
        # Set the number of threads for PyTorch operations to 2 for performance reasons
        torch.set_num_threads(2)
        #Basis evaluation at loss quadrature points
        self.systemnet.device = self.device

    def on_validation_epoch_end(self):
        """
        Perform actions at the end of a validation epoch (hook called by pytorch lightning under the hood).

        Returns:
            None
        """
        print("Validation metric: " +str(self.metric))
