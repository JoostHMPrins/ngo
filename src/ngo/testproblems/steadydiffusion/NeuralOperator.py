# Copyright 2025 Joost Prins

# 3rd Party
import torch
import numpy as np
import pytorch_lightning as pl
import opt_einsum

# Local
from ngo.ml.systemnets import MLP, LBranchNet # , CNN, FNO
from ngo.ml.basisfunctions import TensorizedBasis
from ngo.ml.customlayers import discretize_functions
from ngo.ml.quadrature import GaussLegendreQuadrature, UniformQuadrature

class NeuralOperator(pl.LightningModule):
    """
    Neural Operator class for solving a steady diffusion equation.
    Mainly for the use of Neural Green's Operators (NGOs), and to benchmark these against other models from literature.
    modeltype options are: 'NN', 'DeepONet', 'VarMiON', 'data NGO', 'model NGO', 'FEM' and 'projection'.

    Attributes:
        hparams (dict): Hyperparameters for the model.
        systemnet (torch.nn.Module): The system network.
        basis_test (TensorizedBasis): Basis functions for testing.
        basis_trial (TensorizedBasis): Basis functions for trial.
        Identity (torch.Tensor): Identity matrix, shape (N, N).
    """

    def __init__(self, hparams):
        """
        Initialize the NeuralOperator class.

        Args:
        hparams (dict): Hyperparameters for the model.
        """
        super().__init__()
        self.hparams.update(hparams)
        self.to(self.hparams['device'])
        self.to(self.hparams['dtype'])
        self.compute_quadrature()
        self.init_modeltype()
        self.hparams['N_w_real'] = sum(p.numel() for p in self.parameters()) #True number of trainable parameters
        self.systemnet = self.hparams['systemnet'](self.hparams).to(self.device)
        self.basis_test = TensorizedBasis(self.hparams['test_bases']) 
        self.basis_trial = TensorizedBasis(self.hparams['trial_bases'])
        if self.hparams['Neumannseries']==True:
            self.compute_F_0_A_0() #Neumann preconditioner
        self.Identity = torch.eye(self.hparams['N'], dtype=self.dtype, device=self.device)

    def init_modeltype(self):
        """
        Initialize the model type and set the forward functions and corresponding input and output shapes.

        Returns:
            None
        """
        if self.hparams['modeltype']=='NN':
            self.forwardfunction = self.NN_forward
            self.simforwardfunction = self.NN_simforward
            self.hparams['input_shape'] = (5,)+(self.hparams['Q'])
            self.hparams['output_shape'] = self.hparams['Q_L']
        if self.hparams['modeltype']=='DeepONet':
            self.forwardfunction = self.DeepONet_forward
            self.simforwardfunction = self.DeepONet_simforward
            self.hparams['input_shape'] = (5,)+(self.hparams['Q'])
            self.hparams['output_shape'] = self.hparams['h']
        if self.hparams['modeltype']=='VarMiON':
            self.LBranch_f = LBranchNet(self.hparams, input_dim=self.hparams['Q']**self.hparams['d'], output_dim=self.hparams['N']).to(self.device)
            self.LBranch_eta = LBranchNet(self.hparams, input_dim=2*self.hparams['Q'], output_dim=self.hparams['N']).to(self.device)
            self.LBranch_g = LBranchNet(self.hparams, input_dim=2*self.hparams['Q'], output_dim=self.hparams['N']).to(self.device)
            self.forwardfunction = self.VarMiON_forward
            self.simforwardfunction = self.VarMiON_simforward
            self.hparams['input_shape'] = (1,)+(self.hparams['Q'])
            self.hparams['output_shape'] = (self.hparams['N'],self.hparams['N'])
        if self.hparams['modeltype']=='data NGO':
            self.forwardfunction = self.NGO_forward
            self.simforwardfunction = self.NGO_simforward
            self.hparams['input_shape'] = (1,)+(self.hparams['h'])
            self.hparams['output_shape'] = (self.hparams['N'],self.hparams['N'])           
        if self.hparams['modeltype']=='model NGO':
            self.forwardfunction = self.NGO_forward
            self.simforwardfunction = self.NGO_simforward
            self.hparams['input_shape'] = (1,self.hparams['N'],self.hparams['N'])
            self.hparams['output_shape'] = (self.hparams['N'],self.hparams['N'])
        if self.hparams['modeltype']=='FEM':
            self.forwardfunction = self.FEM_forward
            self.simforwardfunction = self.FEM_simforward
        if self.hparams['modeltype']=='projection':
            self.forwardfunction = self.projection_forward
            self.simforwardfunction = self.projection_simforward

    def discretize_input_functions(self, theta, f, eta_y0, eta_yL, g_x0, g_xL):
        """
        Discretize the input functions.

        Args:
            theta (list of functions): Input function theta, shape len(N_samples).
            f (list of functions): Input function f, shape len(N_samples).
            eta_y0 (list of functions): Input function eta_y0, shape len(N_samples).
            eta_yL (list of functions): Input function eta_yL, shape len(N_samples).
            g_x0 (list of functions): Input function g_x0, shape len(N_samples).
            g_xL (list of functions): Input function g_xL, shape len(N_samples).

        Returns:
            tuple of np.ndarrays: Discretized input functions (theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q),
            each with shape (N_samples, self.xi_i.shape[0]).
        """
        theta_q = discretize_functions(theta, self.xi_Omega)
        theta_x0_q = discretize_functions(theta, self.xi_Gamma_x0)
        theta_xL_q = discretize_functions(theta, self.xi_Gamma_xL)
        f_q = discretize_functions(f, self.xi_Omega)
        eta_y0_q = discretize_functions(eta_y0, self.xi_Gamma_y0)
        eta_yL_q = discretize_functions(eta_yL, self.xi_Gamma_yL)
        g_x0_q = discretize_functions(g_x0, self.xi_Gamma_x0)
        g_xL_q = discretize_functions(g_xL, self.xi_Gamma_xL)
        return theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q
    
    def discretize_output_function(self, u):
        """
        Discretize the output function.

        Args:
            u (list of functions): Output function u, shape len(N_samples).

        Returns:
            np.ndarray: Discretized output function u_d, shape (N_samples, self.xi_Omega_L.shape[0]).
        """
        u_q = discretize_functions(u, self.xi_Omega_L)
        return u_q

    def compute_F(self, theta_q, theta_x0_q, theta_xL_q):
        """
        Compute the system matrix (model NGO or FEM) or mesh-shaped system vector (data NGO) F based on the discretized input functions.
        Optionally adds a stabilization term to ensure positive definiteness of the matrix.

        Args:
            theta_q (np.ndarray):Discretized input function theta, shape (N_samples, self.xi_Omega.shape[0]).
            theta_x0_q (np.ndarray): Discretized iput function theta at x0, shape (N_samples, self.xi_Gamma_x0.shape[0]).
            theta_xL_q (np.ndarray): Discretized iput function theta at xL, shape (N_samples, self.xi_Gamma_xL.shape[0]).

        Returns:
            np.ndarray: Computed matrix F, shape (N_samples, N, N) (for model NGO, FEM) or shape (N_samples, h[0], h[1]) (for data NGO)
        """
        if  self.hparams['modeltype']=='data NGO':
            basis_test = self.basis_test.forward(self.xi_Omega)
            F = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, theta_q)
            F = F.reshape((theta.shape[0],self.hparams['h'][0],self.hparams['h'][1]))
        if  self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM' or self.hparams['modeltype']=='projection':
            gradbasis_test = self.basis_test.grad(self.xi_Omega)
            gradbasis_trial = self.basis_trial.grad(self.xi_Omega)
            basis_test_x0 = self.basis_test.forward(self.xi_Gamma_x0)
            gradbasis_test_x0 = self.basis_test.grad(self.xi_Gamma_x0)
            basis_trial_x0 = self.basis_trial.forward(self.xi_Gamma_x0)
            gradbasis_trial_x0 = self.basis_trial.grad(self.xi_Gamma_x0)
            basis_test_xL = self.basis_test.forward(self.xi_Gamma_xL)
            gradbasis_test_xL = self.basis_test.grad(self.xi_Gamma_xL)
            basis_trial_xL = self.basis_trial.forward(self.xi_Gamma_xL)
            gradbasis_trial_xL = self.basis_trial.grad(self.xi_Gamma_xL)
            F = opt_einsum.contract('q,Nq,qmx,qnx->Nmn', self.w_Omega, theta_q, gradbasis_test, gradbasis_trial)
            F += -opt_einsum.contract('q,qm,x,Nq,qnx->Nmn', self.w_Gamma_x0, basis_test_x0, self.n_x0, theta_x0_q, gradbasis_trial_x0)
            F += -opt_einsum.contract('q,qm,x,Nq,qnx->Nmn', self.w_Gamma_xL, basis_test_xL, self.n_xL, theta_xL_q, gradbasis_trial_xL)
            F += -opt_einsum.contract('q,qn,x,Nq,qmx->Nmn', self.w_Gamma_x0, basis_trial_x0, self.n_x0, theta_x0_q, gradbasis_test_x0)
            F += -opt_einsum.contract('q,qn,x,Nq,qmx->Nmn', self.w_Gamma_xL, basis_trial_xL, self.n_xL, theta_xL_q, gradbasis_test_xL)
            if self.hparams.get('gamma_stabilization',0)!=0:
                F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_x0, basis_test_x0, theta_x0_q, basis_trial_x0)
                F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_xL, basis_test_xL, theta_xL_q, basis_trial_xL)
        return F
    
    def compute_F_0_A_0(self):
        """
        Compute the matrices F_0 and A_0 for the Neumann series.

        Returns:
            None
        """
        theta_q = np.ones(self.w_Omega.shape)
        theta_x0_q = np.ones(self.w_Gamma_x0.shape)
        theta_xL_q = np.ones(self.w_Gamma_xL.shape)
        self.F_0 = torch.tensor(self.compute_F(theta_q, theta_x0_q, theta_xL_q), dtype=self.dtype, device=self.device)
        self.A_0 = torch.linalg.inv(self.F_0)
    
    def compute_d(self, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q):
        """
        Compute the right hand side vector d based on the discretized input functions.

        Args:
            f_q (np.ndarray): Input function f, shape (N_samples, self.xi_Omega.shape[0]).
            eta_y0_q (np.ndarray): Input function eta_y0, shape (N_samples, self.xi_Gamma_y0.shape[0]).
            eta_yL_q (np.ndarray): Input function eta_yL, shape (N_samples, self.xi_Gamma_yL.shape[0]).
            g_x0_q (np.ndarray): Input function g_x0, shape (N_samples, self.xi_Gamma_x0.shape[0]).
            g_xL_q (np.ndarray): Input function g_xL, shape (N_samples, self.xi_Gamma_xL.shape[0]).

        Returns:
            np.ndarray: Computed vector d, shape (N_samples, N).
        """
        basis_test = self.basis_test.forward(self.xi_Omega)
        basis_test_y0 = self.basis_test.forward(self.xi_Gamma_y0)
        basis_test_yL = self.basis_test.forward(self.xi_Gamma_yL)
        basis_test_x0 = self.basis_test.forward(self.xi_Gamma_x0)
        basis_test_xL = self.basis_test.forward(self.xi_Gamma_xL)
        gradbasis_test_x0 = self.basis_test.grad(self.xi_Gamma_x0)
        gradbasis_test_xL = self.basis_test.grad(self.xi_Gamma_xL)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, f_q)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_y0, basis_test_y0, eta_y0_q)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_yL, basis_test_yL, eta_yL_q)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_x0, self.n_x0, gradbasis_test_x0, g_x0_q)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_xL, self.n_xL, gradbasis_test_xL, g_xL_q)
        if self.hparams.get('gamma_stabilization',0)!=0:
            d += -self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_x0, basis_test_x0, g_x0_q)
            d += -self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_xL, basis_test_xL, g_xL_q)                 
        return d
    
    def NN_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q):
        """
        Forward pass for the NN model.
        The boundary conditions are treated as 2D zero tensors with the BC values on the boundaries.

        Args:
            theta_q (torch.tensor):Discretized input function theta, shape (N_samples, self.xi_Omega.shape[0]).
            f_q (torch.tensor): Input function f, shape (N_samples, self.xi_Omega.shape[0]).
            eta_y0_q (torch.tensor): Input function eta_y0, shape (N_samples, self.xi_Gamma_y0.shape[0]).
            eta_yL_q (torch.tensor): Input function eta_yL, shape (N_samples, self.xi_Gamma_yL.shape[0]).
            g_x0_q (torch.tensor): Input function g_x0, shape (N_samples, self.xi_Gamma_x0.shape[0]).
            g_xL_q (torch.tensor): Input function g_xL, shape (N_samples, self.xi_Gamma_xL.shape[0]).

        Returns:
            torch.Tensor: Predicted output u_q_hat, shape (N_samples,)+Q_L.
        """
        theta_q = theta_q.reshape((theta_q.shape[0],)+self.hparams['Q'])
        f_q = f_q.reshape((f_q.shape[0],)+self.hparams['Q'])
        eta_q = torch.zeros((eta_y0_q.shape[0],)+self.hparams['Q'], dtype=self.dtype, device=self.device)
        eta_q[:,:,0] = eta_y0_q
        eta_q[:,:,-1] = eta_yL_q
        g_q = torch.zeros((g_x0_q.shape[0],)+self.hparams['Q'], dtype=self.dtype, device=self.device)
        g_q[:,0,:] = g_x0_q
        g_q[:,-1,:] = g_xL_q
        inputfuncs = torch.stack((theta_q,f_q,eta_q,g_q), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta_q.flatten(-2,-1),f_q.flatten(-2,-1),eta_y0_q,eta_yL_q,g_x0_q,g_xL_q), dim=1)
        u_q_hat = self.systemnet.forward(inputfuncs).reshape((theta_q.shape[0],)+self.hparams['Q_L'])
        return u_q_hat
    
    def NN_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        """
        Simulation forward pass for the NN model (input functions -> solution).

        Args:
            theta (list of functions): Input function theta, shape len(N_samples).
            f (list of functions): Input function f, shape len(N_samples).
            eta_y0 (list of functions): Input function eta_y0, shape len(N_samples).
            eta_yL (list of functions): Input function eta_yL, shape len(N_samples).
            g_x0 (list of functions): Input function g_x0, shape len(N_samples).
            g_xL (list of functions): Input function g_xL, shape len(N_samples).
            u (list of functions): Output function u, shape len(N_samples).

        Returns:
            np.ndarray: Predicted output u_q_hat, shape (N_samples,)+Q_L.
        """
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        theta_q = torch.tensor(theta_q, dtype=self.dtype, device=self.device)
        f_q = torch.tensor(f_q, dtype=self.dtype, device=self.device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.dtype, device=self.device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.dtype, device=self.device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.dtype, device=self.device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.dtype, device=self.device)   
        u_q_hat = self.NN_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q).detach().cpu().numpy()
        return u_q_hat
    
    def DeepONet_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q):
        """
        Forward pass for the DeepONet model.
        The boundary conditions are treated as 2D zero tensors with the BC values on the boundaries.

        Args:
            theta_q (torch.tensor):Discretized input function theta, shape (N_samples, self.xi_Omega.shape[0]).
            f_q (torch.tensor): Input function f, shape (N_samples, self.xi_Omega.shape[0]).
            eta_y0_q (torch.tensor): Input function eta_y0, shape (N_samples, self.xi_Gamma_y0.shape[0]).
            eta_yL_q (torch.tensor): Input function eta_yL, shape (N_samples, self.xi_Gamma_yL.shape[0]).
            g_x0_q (torch.tensor): Input function g_x0, shape (N_samples, self.xi_Gamma_x0.shape[0]).
            g_xL_q (torch.tensor): Input function g_xL, shape (N_samples, self.xi_Gamma_xL.shape[0]).

        Returns:
            torch.Tensor: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        theta_q = theta_q.reshape((theta_q.shape[0],)+self.hparams['Q'])
        f_q = f_q.reshape((f_q.shape[0],)+self.hparams['Q'])
        eta_q = torch.zeros((eta_y0_q.shape[0],)+self.hparams['Q'], dtype=self.dtype, device=self.device)
        eta_q[:,:,0] = eta_y0_q
        eta_q[:,:,-1] = eta_yL_q
        g_q = torch.zeros((g_x0_q.shape[0],)+self.hparams['Q'], dtype=self.dtype, device=self.device)
        g_q[:,0,:] = g_x0_q
        g_q[:,-1,:] = g_xL_q
        inputfuncs = torch.stack((theta_q,f_q,eta_q,g_q), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta_q.flatten(-2,-1),f_q.flatten(-2,-1),eta_y0_q,eta_yL_q,g_x0_q,g_xL_q), dim=1)
        u_n = self.systemnet.forward(inputfuncs).reshape((theta_q.shape[0],self.hparams['N']))
        u_q_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_q_hat
    
    def DeepONet_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        """
        Simulation forward pass for the DeepONet model (input functions -> solution).

        Args:
            theta (list of functions): Input function theta, shape len(N_samples).
            f (list of functions): Input function f, shape len(N_samples).
            eta_y0 (list of functions): Input function eta_y0, shape len(N_samples).
            eta_yL (list of functions): Input function eta_yL, shape len(N_samples).
            g_x0 (list of functions): Input function g_x0, shape len(N_samples).
            g_xL (list of functions): Input function g_xL, shape len(N_samples).
            u (list of functions): Output function u, shape len(N_samples).

        Returns:
            np.ndarray: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        theta_q = torch.tensor(theta_q, dtype=self.dtype, device=self.device)
        f_q = torch.tensor(f_q, dtype=self.dtype, device=self.device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.dtype, device=self.device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.dtype, device=self.device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.dtype, device=self.device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.dtype, device=self.device)   
        u_q_hat = self.DeepONet_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q).detach().cpu().numpy()
        return u_q_hat
    
    def VarMiON_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q):
        """
        Forward pass for the VarMiON model.
        The boundary conditions are treated as 2D zero tensors with the BC values on the boundaries.

        Args:
            theta_q (torch.tensor):Discretized input function theta, shape (N_samples, self.xi_Omega.shape[0]).
            f_q (torch.tensor): Input function f, shape (N_samples, self.xi_Omega.shape[0]).
            eta_y0_q (torch.tensor): Input function eta_y0, shape (N_samples, self.xi_Gamma_y0.shape[0]).
            eta_yL_q (torch.tensor): Input function eta_yL, shape (N_samples, self.xi_Gamma_yL.shape[0]).
            g_x0_q (torch.tensor): Input function g_x0, shape (N_samples, self.xi_Gamma_x0.shape[0]).
            g_xL_q (torch.tensor): Input function g_xL, shape (N_samples, self.xi_Gamma_xL.shape[0]).

        Returns:
            torch.Tensor: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        eta_q = torch.zeros((eta_y0_q.shape[0],2*eta_y0_q.shape[1]), dtype=self.dtype, device=self.device)
        eta_q[:,:eta_y0_q.shape[1]] = eta_y0_q
        eta_q[:,eta_y0_q.shape[1]:] = eta_yL_q
        g_q = torch.zeros((g_x0_q.shape[0],2*g_x0_q.shape[1]), dtype=self.dtype, device=self.device)
        g_q[:,:g_x0.shape[1]] = g_x0_q
        g_q[:,g_x0.shape[1]:] = g_xL_q
        systemnet = self.systemnet.forward(theta_q)
        LBranch = self.LBranch_f.forward(f_q) + self.LBranch_eta.forward(eta_q) + self.LBranch_g.forward(g_q)
        u_n = opt_einsum.contract('nij,nj->ni', systemnet, LBranch)
        u_q_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_q_hat
    
    def VarMiON_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        """
        Simulation forward pass for the VarMiON model (input functions -> solution).

        Args:
            theta (list of functions): Input function theta, shape len(N_samples).
            f (list of functions): Input function f, shape len(N_samples).
            eta_y0 (list of functions): Input function eta_y0, shape len(N_samples).
            eta_yL (list of functions): Input function eta_yL, shape len(N_samples).
            g_x0 (list of functions): Input function g_x0, shape len(N_samples).
            g_xL (list of functions): Input function g_xL, shape len(N_samples).
            u (list of functions): Output function u, shape len(N_samples).

        Returns:
            np.ndarray: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        theta_q = torch.tensor(theta_q, dtype=self.dtype, device=self.device)
        f_q = torch.tensor(f_q, dtype=self.dtype, device=self.device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.dtype, device=self.device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.dtype, device=self.device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.dtype, device=self.device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.dtype, device=self.device)   
        u_q_hat = self.VarMiON_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q).detach().cpu().numpy()
        return u_q_hat

    def NGO_forward(self, scaling, F, d):
        """
        Forward pass for the NGO model.

        Args:
            scaling (torch.Tensor): Scaling factor, shape (N_samples,).
            F (torch.Tensor): Input matrix F, shape (N_samples, N, N) (for model NGO) or (N_samples, h[0], h[1]) (for data NGO).
            d (torch.Tensor): Input vector d, shape (N_samples, N).

        Returns:
            torch.Tensor: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        if self.hparams.get('scale_equivariance',False)==True:
            F = F/scaling[:,None,None]
        if self.hparams.get('Neumannseries', False)==False:
            A = self.systemnet.forward(F)
        if self.hparams.get('Neumannseries', False)==True:
            T = torch.zeros(F.shape, dtype=self.dtype, device=self.device)
            Ti = self.Identity
            T1 = -F@self.A_0 + self.Identity
            for i in range(0, self.hparams['Neumannseries_order']):
                Ti = T1@Ti
                T = T + Ti
            A = self.A_0@(self.Identity + T + self.systemnet.forward(T1))
        if self.hparams.get('scale_equivariance',False)==True:
            A = A/scaling[:,None,None]        
        u_n = opt_einsum.contract('nij,nj->ni', A, d)
        u_q_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_q_hat
    
    def NGO_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        """
        Simulation forward pass for the NGO model (input functions -> solution).

        Args:
            theta (list of functions): Input function theta, shape len(N_samples).
            f (list of functions): Input function f, shape len(N_samples).
            eta_y0 (list of functions): Input function eta_y0, shape len(N_samples).
            eta_yL (list of functions): Input function eta_yL, shape len(N_samples).
            g_x0 (list of functions): Input function g_x0, shape len(N_samples).
            g_xL (list of functions): Input function g_xL, shape len(N_samples).
            u (list of functions): Output function u, shape len(N_samples).

        Returns:
            np.ndarray: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        F = torch.tensor(self.compute_F(theta_q, theta_x0_q, theta_xL_q), dtype=self.dtype, device=self.device)
        d = torch.tensor(self.compute_d(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q), dtype=self.dtype, device=self.device)
        scaling = torch.tensor(np.sum(self.w_Omega[None,:]*theta_q, axis=-1), dtype=self.dtype, device=self.device)
        u_q_hat = self.NGO_forward(scaling, F, d).detach().cpu().numpy()
        return u_q_hat
    
    def FEM_forward(self, F, d):
        """
        Forward pass for the FEM model.

        Args:
            F (torch.Tensor): Input matrix F, shape (N_samples, N, N).
            d (torch.Tensor): Input vector d, shape (N_samples, N).

        Returns:
            torch.Tensor: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        if self.hparams.get('Neumannseries', False)==False:
            K_inv = torch.linalg.pinv(F)
        if self.hparams.get('Neumannseries', False)==True:
            T = torch.zeros(F.shape, dtype=self.dtype, device=self.device)
            Ti = self.Identity
            T1 = -F@self.A_0 + self.Identity
            for i in range(0, self.hparams['Neumannseries_order']):
                Ti = T1@Ti
                T = T + Ti
            K_inv = self.A_0@(self.Identity + T)
        u_n = opt_einsum.contract('nij,nj->ni', K_inv, d)
        u_q_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_q_hat
    
    def FEM_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        """
        Simulation forward pass for the FEM model (input functions -> solution).

        Args:
            theta (list of functions): Input function theta, shape len(N_samples).
            f (list of functions): Input function f, shape len(N_samples).
            eta_y0 (list of functions): Input function eta_y0, shape len(N_samples).
            eta_yL (list of functions): Input function eta_yL, shape len(N_samples).
            g_x0 (list of functions): Input function g_x0, shape len(N_samples).
            g_xL (list of functions): Input function g_xL, shape len(N_samples).
            u (list of functions): Output function u, shape len(N_samples).

        Returns:
            np.ndarray: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        F = torch.tensor(self.compute_F(theta_q, theta_x0_q, theta_xL_q), dtype=self.dtype, device=self.device)
        d = torch.tensor(self.compute_d(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q), dtype=self.dtype, device=self.device)
        u_q_hat = self.FEM_forward(F, d).detach().cpu().numpy()
        return u_q_hat
    
    def projection_forward(self, u):
        """
        Forward pass for the projection model.

        Args:
            u (list of functions): Output function u, shape (N_samples, ...).

        Returns:
            torch.Tensor: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        u_q = discretize_functions(u, self.xi_Omega)
        basis_trial = self.basis_trial.forward(self.xi_Omega)
        u_w = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_trial, u_q)
        M = opt_einsum.contract('q,qm,qn->mn', self.w_Omega, basis_trial, basis_trial)
        M_inv = np.linalg.pinv(M)
        u_n = opt_einsum.contract('mn,Nm->Nn', M_inv, u_w)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat
    
    def projection_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        """
        Simulation forward pass for the projection model (input functions -> solution).

        Args:
            u (list of functions): Output function u, shape len(N_samples).
            
        Returns:
            np.ndarray: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        u_hat = self.projection_forward(u)
        return u_hat
    
    def forward(self, *args):
        """
        Training forward pass for the model, set in the function self.init_modeltype(),
        based on the selected hparams['modeltype'].

        Args:
            *args: Variable length argument list.

        Returns:
            torch.Tensor: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        u_q_hat = self.forwardfunction(*args)
        return u_q_hat
    
    def simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, x, u):
        """
        Simulation forward pass for the model (input functions -> solution),
        set in the function self.init_modeltype(), based on the selected hparams['modeltype'].

        Args:
            theta (list of functions): Input function theta, shape len(N_samples).
            f (list of functions): Input function f, shape len(N_samples).
            eta_y0 (list of functions): Input function eta_y0, shape len(N_samples).
            eta_yL (list of functions): Input function eta_yL, shape len(N_samples).
            g_x0 (list of functions): Input function g_x0, shape len(N_samples).
            g_xL (list of functions): Input function g_xL, shape len(N_samples).
            u (list of functions): Output function u, shape len(N_samples).
            x (np.ndarray): (Quadrature) points to evaluate the solution u, shape (N_points,dimension).
        Returns:
            np.ndarray: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        self.psix = torch.tensor(self.basis_trial.forward(x), dtype=self.dtype, device=self.device)
        u_hat = self.simforwardfunction(theta, f, eta_y0, eta_yL, g_x0, g_xL, u)
        return u_hat
    
    def compute_quadrature(self):
        """
        Compute the quadrature points and weights, and the geometry outward normal.

        Returns:
            None
        """
        #Quadrature
        if self.hparams['quadrature']=='uniform':
            quad_Omega = UniformQuadrature(Q=self.hparams['Q'])
            quad_Gamma_x = UniformQuadrature(Q=[self.hparams['Q'][1]])
            quad_Gamma_y = UniformQuadrature(Q=[self.hparams['Q'][0]])
        if self.hparams['quadrature']=='Gauss-Legendre':
            quad_Omega = GaussLegendreQuadrature(Q=self.hparams['Q'], n_elements=self.hparams['n_elements'])
            quad_Gamma_x = GaussLegendreQuadrature(Q=[self.hparams['Q'][1]], n_elements=[self.hparams['n_elements'][1]])
            quad_Gamma_y = GaussLegendreQuadrature(Q=[self.hparams['Q'][0]], n_elements=[self.hparams['n_elements'][0]])
        self.w_Omega = quad_Omega.w
        self.xi_Omega = quad_Omega.xi
        self.w_Gamma_x0 = quad_Gamma_x.w
        self.xi_Gamma_x0 = np.zeros((self.hparams['Q'][1],self.hparams['d']))
        self.xi_Gamma_x0[:,1] = quad_Gamma_x.xi[:,0]
        self.w_Gamma_xL = quad_Gamma_x.w
        self.xi_Gamma_xL = np.ones((self.hparams['Q'][1],self.hparams['d']))
        self.xi_Gamma_xL[:,1] = quad_Gamma_x.xi[:,0]
        self.w_Gamma_y0 = quad_Gamma_y.w
        self.xi_Gamma_y0 = np.zeros((self.hparams['Q'][0],self.hparams['d']))
        self.xi_Gamma_y0[:,0] = quad_Gamma_y.xi[:,0]
        self.w_Gamma_yL = quad_Gamma_y.w
        self.xi_Gamma_yL = np.ones((self.hparams['Q'][0],self.hparams['d']))
        self.xi_Gamma_yL[:,0] = quad_Gamma_y.xi[:,0]

        #Loss quadrature
        if self.hparams['quadrature_L']=='uniform':
            quadrature_L = UniformQuadrature(Q=self.hparams['Q_L'])
        if self.hparams['quadrature_L']=='Gauss-Legendre':
            quadrature_L = GaussLegendreQuadrature(Q=self.hparams['Q_L'], n_elements=self.hparams['n_elements_L'])
        self.xi_Omega_L = quadrature_L.xi
        self.w_Omega_L = quadrature_L.w

        #Outward normal
        self.n_y0 = np.array([0,-1])
        self.n_yL = np.array([0,1])
        self.n_x0 = np.array([-1,0])
        self.n_xL = np.array([1,0])

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
            loss += self.hparams['solution_loss'](self.w_Omega_L, u_hat, u)
        if self.hparams.get('matrix_loss',None)!=None:
            F = inputs[1]
            if self.hparams.get('Neumannseries', False)==False:
                A_hat = self.systemnet.forward(F)
            if self.hparams.get('Neumannseries', False)==True:
                T = torch.zeros(F.shape, dtype=self.dtype, device=self.device)
                Ti = self.Identity
                T1 = - (F - self.F_0)@self.A_0
                for i in range(0, self.hparams['Neumannseries_order']):
                    Ti = T1@Ti
                    T = T + Ti
                A_hat = self.A_0@(self.Identity + T + self.systemnet.forward(T1))
            loss += self.hparams['matrix_loss'](torch.matmul(F, torch.matmul(A_hat,F)), F)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Definition of the validation step (called under the hood by pytorch lightning).

        Args:
            train_batch (tuple): Batch of training data.
            batch_idx (int): Index of the batch.

        Returns:
            None
        """
        inputs = val_batch[:-1]
        u = val_batch[-1]
        u_hat = self.forward(*inputs)
        loss = 0
        if self.hparams.get('solution_loss',None)!=None:
            loss += self.hparams['solution_loss'](self.w_Omega_L, u_hat, u)
        if self.hparams.get('matrix_loss',None)!=None:
            F = inputs[1]
            if self.hparams.get('Neumannseries',False)==False:
                A_hat = self.systemnet.forward(F)
            if self.hparams.get('Neumannseries',False)==True:
                T = torch.zeros(F.shape, dtype=self.dtype, device=self.device)
                Ti = self.Identity
                T1 = - (F - self.F_0)@self.A_0
                for i in range(0, self.hparams['Neumannseries_order']):
                    Ti = T1@Ti
                    T = T + Ti
                A_hat = self.A_0@(self.Identity + T + self.systemnet.forward(T1))
            loss += self.hparams['matrix_loss'](torch.matmul(F, torch.matmul(A_hat,F)), F)
        self.metric = self.hparams['metric'](self.w_Omega_L, u_hat, u)
        self.log('val_loss', loss)
        self.log('metric', self.metric)

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
        torch.set_num_threads(2)
        #Basis evaluation at loss quadrature points
        self.psix = torch.tensor(self.basis_trial.forward(self.xi_Omega_L), dtype=self.dtype, device=self.device).to(self.device)
        self.w_Omega_L = torch.tensor(self.w_Omega_L).to(self.device)
        self.systemnet.device = self.device

    def on_validation_epoch_end(self):
        """
        Perform actions at the end of a validation epoch (hook called by pytorch lightning under the hood).

        Returns:
            None
        """
        print("Training metric: " +str(self.metric))