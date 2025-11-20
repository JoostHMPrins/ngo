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
from ngo.ml.customlayers import Neumann_preconditioner_forward, discretize_functions

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
        # Initialize test and trial basis functions
        self.psi = TensorProductBasis(self.hparams['test_bases']) 
        self.phi = TensorProductBasis(self.hparams['trial_bases'])
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
            self.hparams['input_shape'] = (4,)+(self.hparams['Q'])
            self.hparams['output_shape'] = self.hparams['Q_L']
        if self.hparams['modeltype']=='DeepONet':
            self.forwardfunction = self.DeepONet_forward
            self.simforwardfunction = self.DeepONet_simforward
            self.hparams['input_shape'] = (4,)+(self.hparams['Q'])
            self.hparams['output_shape'] = self.hparams['h']
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
        theta_q = theta.forward(self.Xi_Omega)
        theta_x0_q = theta.forward(self.Xi_Gamma_x0)
        theta_xL_q = theta.forward(self.Xi_Gamma_xL)
        f_q = f.forward(self.Xi_Omega)
        eta_y0_q = eta_y0.forward(self.Xi_Gamma_y0)
        eta_yL_q = eta_yL.forward(self.Xi_Gamma_yL)
        g_x0_q = g_x0.forward(self.Xi_Gamma_x0)
        g_xL_q = g_xL.forward(self.Xi_Gamma_xL)
        return theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q
    
    def discretize_output_function(self, u):
        """
        Discretize the output function.

        Args:
            u (list of functions): Output function u, shape len(N_samples).

        Returns:
            np.ndarray: Discretized output function u_d, shape (N_samples, self.Xi_Omega_L.shape[0]).
        """
        u_q = u.forward(self.Xi_Omega_L)
        return u_q
    
    def assemble_F(self, theta):
        if  self.hparams['modeltype']=='data NGO':
            F = assemble_L(self.psi, theta, self.W_Omega, self.Xi_Omega, derivative_orders=((0,0),(0,0)))
        if  self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM' or self.hparams['modeltype']=='projection':
            # Volume term
            T_i = assemble_T(theta, self.psi, self.phi, self.W_Omega, self.Xi_Omega, derivative_orders=((0,0),(1,1),(1,1)))
            # Boundary term at x0 (coming from Neumann complement)
            T_i = subtract_tuples(T_i, assemble_T(theta, self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0),(0,0),(1,0))))
            # Boundary term at xL (coming from Neumann complement)
            T_i = subtract_tuples(T_i, assemble_T(theta, self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0),(0,0),(1,0))))
            # Boundary term at x0 (coming from Dirichlet complement)
            T_i = subtract_tuples(T_i, assemble_T(theta, self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0),(1,0),(0,0))))
            # Boundary term at xL (coming from Dirichlet complement)
            T_i = subtract_tuples(T_i, assemble_T(theta, self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0),(1,0),(0,0))))
            if self.hparams.get('gamma_stabilization',0)!=0:
                # Stabilization term at x0
                T_i = add_tuples(T_i, self.hparams['gamma_stabilization']*assemble_T(theta, self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0),(0,0),(0,0))))
                # Stabilization term at xL
                T_i = add_tuples(T_i, self.hparams['gamma_stabilization']*assemble_T(theta, self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0),(0,0),(0,0))))
            F_i = kron_vecT(theta.u_m, T_i)
            F = kron(F_i)
        return F

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
            basis_test = self.psi.forward(self.xi_Omega)
            F = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, theta_q)
            F = F.reshape((theta_q.shape[0],self.hparams['h'][0],self.hparams['h'][1]))
        if  self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM' or self.hparams['modeltype']=='projection':
            gradbasis_test = self.psi.grad(self.xi_Omega)
            gradbasis_trial = self.phi.grad(self.xi_Omega)
            basis_test_x0 = self.psi.forward(self.xi_Gamma_x0)
            gradbasis_test_x0 = self.psi.grad(self.xi_Gamma_x0)
            basis_trial_x0 = self.phi.forward(self.xi_Gamma_x0)
            gradbasis_trial_x0 = self.phi.grad(self.xi_Gamma_x0)
            basis_test_xL = self.psi.forward(self.xi_Gamma_xL)
            gradbasis_test_xL = self.psi.grad(self.xi_Gamma_xL)
            basis_trial_xL = self.phi.forward(self.xi_Gamma_xL)
            gradbasis_trial_xL = self.phi.grad(self.xi_Gamma_xL)
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
        # Volume term
        B_i = assemble_B(self.psi, self.phi, self.W_Omega, self.Xi_Omega, derivative_orders=((0,0),(1,1),(1,1)))
        # Boundary term at x0 (coming from Neumann complement)
        B_i = subtract_tuples(B_i, assemble_T(self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0),(0,0),(1,0))))
        # Boundary term at xL (coming from Neumann complement)
        B_i = subtract_tuples(B_i, assemble_T(self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0),(0,0),(1,0))))
        # Boundary term at x0 (coming from Dirichlet complement)
        B_i = subtract_tuples(B_i, assemble_T(self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0),(1,0),(0,0))))
        # Boundary term at xL (coming from Dirichlet complement)
        B_i = subtract_tuples(B_i, assemble_T(self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0),(1,0),(0,0))))
        if self.hparams.get('gamma_stabilization',0)!=0:
            # Stabilization term at x0
            B_i = add_tuples(B_i, self.hparams['gamma_stabilization']*assemble_T(self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0),(0,0),(0,0))))
            # Stabilization term at xL
            B_i = add_tuples(B_i, self.hparams['gamma_stabilization']*assemble_T(self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0),(0,0),(0,0))))
        self.F_0 = kron(B_i)
        A_0_i = kron_inv(B_i)
        self.A_0 = kron(A_0_i)
    
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
        basis_test = self.psi.forward(self.xi_Omega)
        basis_test_y0 = self.psi.forward(self.xi_Gamma_y0)
        basis_test_yL = self.psi.forward(self.xi_Gamma_yL)
        basis_test_x0 = self.psi.forward(self.xi_Gamma_x0)
        basis_test_xL = self.psi.forward(self.xi_Gamma_xL)
        gradbasis_test_x0 = self.psi.grad(self.xi_Gamma_x0)
        gradbasis_test_xL = self.psi.grad(self.xi_Gamma_xL)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, f_q)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_y0, basis_test_y0, eta_y0_q)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_yL, basis_test_yL, eta_yL_q)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_x0, self.n_x0, gradbasis_test_x0, g_x0_q)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_xL, self.n_xL, gradbasis_test_xL, g_xL_q)
        if self.hparams.get('gamma_stabilization',0)!=0:
            d += -self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_x0, basis_test_x0, g_x0_q)
            d += -self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_xL, basis_test_xL, g_xL_q)                 
        return d
    
    def assemble_d(self, f, eta_y0, eta_yL, g_x0, g_xL):
        d = assemble_L(self.psi, f, self.W_Omega, self.Xi_Omega, derivative_orders=(0,0))
        d += assemble_L(self.psi, eta_y0, self.W_Gamma_y0, self.Xi_Gamma_y0, derivative_orders=(0,0))
        d += assemble_L(self.psi, eta_yL, self.W_Gamma_yL, self.Xi_Gamma_yL, derivative_orders=(0,0))
        d += assemble_L(self.psi, g_x0, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=(1,0)) # addition because of minus sign of boundary normal sign at x0
        d -= assemble_L(self.psi, g_xL, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=(1,0))
        if self.hparams.get('gamma_stabilization',0)!=0:
            d -= self.hparams['gamma_stabilization']*assemble_L(self.psi, g_x0, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=(0,0))
            d -= self.hparams['gamma_stabilization']*assemble_L(self.psi, g_xL, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=(0,0))
        d = d.flatten(start_dim=1)
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
        g_x0_q = g_x0_q.reshape((g_x0_q.shape[0],)+(self.hparams['Q'][1],))
        g_xL_q = g_xL_q.reshape((g_xL_q.shape[0],)+(self.hparams['Q'][1],))  
        g_q = torch.zeros((g_x0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        g_q[:,0,:] = g_x0_q
        g_q[:,-1,:] = g_xL_q
        inputfuncs = torch.stack((theta_q,f_q,eta_q,g_q), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta_q.flatten(-2,-1),f_q.flatten(-2,-1),eta_y0_q,eta_yL_q,g_x0_q,g_xL_q), dim=1)
        u_q_hat = self.systemnet.forward(inputfuncs).reshape((theta_q.shape[0],)+(np.prod(self.hparams['Q_L']),))
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
        theta_q = theta_q.reshape((theta_q.shape[0],)+(self.hparams['Q']))
        f_q = f_q.reshape((f_q.shape[0],)+(self.hparams['Q']))
        eta_y0_q = eta_y0_q.reshape((eta_y0_q.shape[0],)+(self.hparams['Q'][0],))
        eta_yL_q = eta_yL_q.reshape((eta_yL_q.shape[0],)+(self.hparams['Q'][0],))
        eta_q = torch.zeros((eta_y0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        eta_q[:,:,0] = eta_y0_q
        eta_q[:,:,-1] = eta_yL_q
        g_x0_q = g_x0_q.reshape((g_x0_q.shape[0],)+(self.hparams['Q'][1],))
        g_xL_q = g_xL_q.reshape((g_xL_q.shape[0],)+(self.hparams['Q'][1],))  
        g_q = torch.zeros((g_x0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
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
        g_q[:,:g_x0_q.shape[1]] = g_x0_q
        g_q[:,g_x0_q.shape[1]:] = g_xL_q
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
            A_hat = self.systemnet.forward(F)
        if self.hparams.get('Neumannseries', False)==True:
            A_hat = Neumann_preconditioner_forward(self, F)
        if self.hparams.get('scale_equivariance',False)==True:
            A_hat = A_hat/scaling[:,None,None]        
        u_n_hat = opt_einsum.contract('nij,nj->ni', A_hat, d)
        # u_q_hat = opt_einsum.contract('Nn,qn->Nq', u_n_hat, self.psix)
        u_q_hat = self.phi.forward(u_n_hat, self.Xi_Omega_L)
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
            F_inv = torch.linalg.pinv(F)
        if self.hparams.get('Neumannseries', False)==True:
            F_inv = Neumann_preconditioner_forward(self, F)
        u_n = opt_einsum.contract('nij,nj->ni', F_inv, d)
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
        L2 projection of the solution onto the trial basis functions.

        Args:
            u (list of functions): Output function u, shape (N_samples, ...).

        Returns:
            torch.Tensor: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        M = assemble_B(self.phi, self.phi, self.W_Omega, self.Xi_Omega, derivative_orders=((0,0),(0,0)))
        d = assemble_L(self.phi, u, self.W_Omega, self.Xi_Omega, derivative_orders=(0,0))
        u_m_hat = kron_solve(M, d)      
        return u_m_hat
    
    def projection_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        """
        Simulation forward pass for the projection model (input functions -> solution).

        Args:
            u (list of functions): Output function u, shape len(N_samples).
            
        Returns:
            np.ndarray: Predicted output u_q_hat, shape (N_samples,)+(x.shape[0],).
        """
        u_hat = self.projection_forward(u).detach().cpu().numpy()
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
    
    def simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, X, u):
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
        theta_q = discretize_functions(theta, self.xi_Omega)
        theta_x0_q = discretize_functions(theta, self.xi_Gamma_x0)
        theta_xL_q = discretize_functions(theta, self.xi_Gamma_xL)
        F = torch.tensor(self.compute_F(theta_q, theta_x0_q, theta_xL_q), dtype=self.hparams['dtype'], device=self.device)
        if self.hparams.get('scaling_equivariance',False)==True:
            scaling = torch.tensor(torch.sum(self.w_Omega[None,:]*theta_q, axis=-1), dtype=self.hparams['dtype'], device=self.device)
            F = F/scaling[:,None,None]
        T1 = -F@self.A_0 + self.Identity
        eigvals = torch.linalg.eigvals(T1)
        spectralradius = torch.amax(torch.real((eigvals*torch.conj(eigvals))**(1/2)), dim=-1)
        return spectralradius
    
    def compute_quadrature(self):
        """
        Compute the quadrature points and weights, and the geometry outward normal.

        Returns:
            None
        """
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
            loss += self.hparams['solution_loss'](self.w_Omega_L, u_hat, u)
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
                A_hat = Neumann_preconditioner_forward(self, F)
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
        # self.psix = torch.tensor(self.phi.forward(self.Xi_Omega_L), dtype=self.dtype, device=self.device).to(self.device)
        self.w_Omega_L = torch.tensor(self.w_Omega_L).to(self.device)
        self.systemnet.device = self.device

    def on_validation_epoch_end(self):
        """
        Perform actions at the end of a validation epoch (hook called by pytorch lightning under the hood).

        Returns:
            None
        """
        print("Training metric: " +str(self.metric))
