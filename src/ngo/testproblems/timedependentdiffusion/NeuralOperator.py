# Copyright 2025 Joost Prins

# 3rd Party
from pyexpat import model
import torch
import numpy as np
import pytorch_lightning as pl
import opt_einsum
import os

# Local
from ngo.numericalanalysis.basisfunctions import TensorProductBasis
from ngo.numericalanalysis.quadrature import GaussLegendreQuadrature, UniformQuadrature
from ngo.numericalanalysis.linalg import *
from ngo.ml.systemnets import MLP, LBranchNet, FNO
from ngo.ml.customlayers import Neumann_preconditioner_forward, M_norm_scaling, conserve_mass, conserve_energy, conserve_mass_and_energy, two_norm_power_iteration
from ngo.trainingdata.GRF import UnitFunction

class NeuralOperator(pl.LightningModule):
    """
    Neural Operator class for solving a time-dependent diffusion equation.
    Mainly for the use of Neural Green's Operators (NGOs), and to benchmark these against other models from literature.
    modeltype options are: 'NN', 'DeepONet', 'VarMiON', 'data NGO', 'model NGO', 'FEM' and 'projection'.

    Attributes:
        hparams (dict): Hyperparameters for the model.
        systemnet (torch.nn.Module): The system network.
        basis_test (TensorizedBasis): Test basis functions.
        basis_trial (TensorizedBasis): Trial basis functions.
        Identity (torch.Tensor): Identity matrix, shape (N, N)
        M_i (list of torch.Tensor): List of mass matrices per dimension.
        M_lr_i (list of torch.Tensor): List of left-right mass matrices per dimension.
        M_rr_i (list of torch.Tensor): List of right-right mass matrices per dimension.
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
        self = self.to(self.hparams['dtype'])
        #compute_quadrature and quadrature
        self.compute_quadrature()
        self.timestep = int(0)
        #Bases
        self.psi = TensorProductBasis(self.hparams['test_bases']) 
        self.phi = TensorProductBasis(self.hparams['trial_bases'])
        # Scale bases to Dt
        self.psi.bases[0].L = self.hparams['Dt']
        self.phi.bases[0].L = self.hparams['Dt']
        self.psi.bases[0].Dx = 0.0
        self.phi.bases[0].Dx = 0.0
        # Assemble mass matrices
        self.M_i = self.assemble_M_i()
        self.M_lr_i = self.assemble_M_lr_i()
        self.M_rr_i = self.assemble_M_rr_i()
        #Model type
        self.init_modeltype()
        #System net
        self.hparams['N_w_real'] = sum(p.numel() for p in self.parameters())
        self.systemnet = self.hparams['systemnet'](self.hparams).to(self.device)
        
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
            self.hparams['output_shape'] = self.hparams['N_i']
        if self.hparams['modeltype']=='VarMiON':
            # Define linear branch networks for each linear input function
            self.LBranch_f = LBranchNet(self.hparams, input_dim=self.hparams['Q'][0]*self.hparams['Q'][1]*self.hparams['Q'][2], output_dim=self.hparams['N']).to(self.device)
            self.LBranch_eta = LBranchNet(self.hparams, input_dim=2*self.hparams['Q'][0]*self.hparams['Q'][1], output_dim=self.hparams['N']).to(self.device)
            self.LBranch_g = LBranchNet(self.hparams, input_dim=2*self.hparams['Q'][0]*self.hparams['Q'][2], output_dim=self.hparams['N']).to(self.device)
            self.LBranch_u0 = LBranchNet(self.hparams, input_dim=self.hparams['Q'][1]*self.hparams['Q'][2], output_dim=self.hparams['N']).to(self.device)
            self.forwardfunction = self.VarMiON_forward
            self.simforwardfunction = self.VarMiON_simforward
            self.hparams['input_shape'] = (1,)+(self.hparams['Q'])
            self.hparams['output_shape'] = (self.hparams['N'],self.hparams['N'])
            # initialize power iteration vector and F_0 for M-norm scaling
            self.F_0 = torch.ones((1,)+self.hparams['Q'], dtype=self.dtype, device=self.device)
            x_0 = torch.randn(int(self.hparams['N']/2)) #Random initial vector for power iteration
            x_0 = x_0/torch.sum(x_0**2)**(1/2)
            self.x_0 = torch.tensor(x_0, dtype=self.dtype, device=self.device)
            #If a previous x_0 exists for a trained model, load it
            if os.path.exists(self.hparams['logdir']+'/x_0.pt'):
                print('loading x_0 from file')
                self.x_0 = torch.load(self.hparams['logdir']+'/x_0.pt').to(self.device) 
            self.Identity = torch.eye(self.hparams['N'], dtype=self.dtype, device=self.device) 
        if self.hparams['modeltype']=='data NGO':
            self.forwardfunction = self.NGO_forward
            self.simforwardfunction = self.NGO_simforward
            self.hparams['input_shape'] = (1,)+self.hparams['N_i']
            self.hparams['output_shape'] = (self.hparams['N'],self.hparams['N'])
            # assemble mass matrices needed for norm scaling computations
            self.M_lr = kron(self.M_lr_i)[0]/self.hparams['Dt']
            self.M_rr = kron(self.M_rr_i)[0]/self.hparams['Dt']
            M_rr_22 = self.M_rr[int(self.hparams['N']/2):,int(self.hparams['N']/2):]
            Lambda_22, T_22 = torch.linalg.eig(M_rr_22)
            self.M_rr22_05 = T_22 @ torch.diag(Lambda_22**0.5) @ T_22.T
            self.M_rr22_05 = torch.tensor(self.M_rr22_05, dtype=self.dtype, device=self.device)
            self.M_rr22_05_inv = T_22 @ torch.diag(1/Lambda_22**0.5) @ T_22.T
            self.M_rr22_05_inv = torch.tensor(self.M_rr22_05_inv, dtype=self.dtype, device=self.device)
            self.assemble_Neumann_preconditioner()
            self.Mnorm_0 = 1
             #Random initial vector for power iteration
            x_0 = torch.randn(int(self.hparams['N']/2))
            x_0 = x_0/torch.sum(x_0**2)**(1/2)
            self.x_0 = torch.tensor(x_0, dtype=self.dtype, device=self.device)
            #If a previous x_0 exists for a trained model, load it
            if os.path.exists(self.hparams['logdir']+'/x_0.pt'):
                print('loading x_0 from file')  
                self.x_0 = torch.load(self.hparams['logdir']+'/x_0.pt').to(self.device)
            self.Identity = torch.eye(self.hparams['N'], dtype=self.dtype, device=self.device) 
        if self.hparams['modeltype']=='model NGO':
            self.forwardfunction = self.NGO_forward
            self.simforwardfunction = self.NGO_simforward
            self.hparams['input_shape'] = (1,self.hparams['N'],self.hparams['N'])
            self.hparams['output_shape'] = (self.hparams['N'],self.hparams['N'])
            #Assemble Neumann preconditioner if needed
            if self.hparams['Neumannseries']==True:
                self.assemble_Neumann_preconditioner()
            self.Identity = torch.eye(self.hparams['N'], dtype=self.dtype, device=self.device)
        if self.hparams['modeltype']=='FEM':
            self.forwardfunction = self.FEM_forward
            self.simforwardfunction = self.FEM_simforward
            #Assemble Neumann preconditioner if needed
            if self.hparams['Neumannseries']==True:
                self.assemble_Neumann_preconditioner()
            self.Identity = torch.eye(self.hparams['N'], dtype=self.dtype, device=self.device)
        if self.hparams['modeltype']=='projection':
            self.forwardfunction = self.projection_forward
            self.simforwardfunction = self.projection_simforward
            #Assemble Neumann preconditioner if needed
            if self.hparams['Neumannseries']==True:
                self.assemble_Neumann_preconditioner()
            self.Identity = torch.eye(self.hparams['N'], dtype=self.dtype, device=self.device)

    def discretize_input_functions(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        """
        Discretize the input functions.

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u0 (see manufacturedsolutions.py)

        Returns:
            tuple of torch.Tensor: Discretized input functions (theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q).
            Shapes: all (B, Q_0, ..., Q_{d-1}) (depending on used domain/boundary quadrature).
        """
        theta_q = theta.forward(self.Xi_OmegaT)
        f_q = f.forward(self.Xi_OmegaT)
        eta_y0_q = eta_y0.forward(self.Xi_Gamma_y0)
        eta_yL_q = eta_yL.forward(self.Xi_Gamma_yL)
        g_x0_q = g_x0.forward(self.Xi_Gamma_x0)
        g_xL_q = g_xL.forward(self.Xi_Gamma_xL)
        u0_t0_q = u0.forward(self.Xi_Gamma_t0)
        return theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q
    
    def discretize_output_function(self, u):
        """
        Discretize the output function.

        Args:
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Discretized output function u_q, shape (B, Q_0, ..., Q_{d-1}).
        """
        u_q = u.forward(self.Xi_OmegaT_L)
        return u_q
    
    def project_input_function(self, u):
        """
        Project the input function onto the test basis.

        Args:
            u (ManufacturedSolution / see manufacturedsolutions.py)

        Returns:
            torch.Tensor: Projected input function coefficients, shape (B, N_0, ..., N_{d-1}).
        """
        d = assemble_L(self.psi, u, self.W_OmegaT, self.Xi_OmegaT, derivative_orders=(0,0,0))
        u_m_hat = kron_solve(self.M_i, d)      
        return u_m_hat
    
    def project_output_function(self, u):
        """
        Project the input function onto the trial basis.

        Args:
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Projected input function coefficients, shape (B, N_0, ..., N_{d-1}).
        """
        self.M_i = self.assemble_M_i()
        d = assemble_L(self.phi, u, self.W_OmegaT, self.Xi_OmegaT, derivative_orders=(0,0,0))
        u_m_hat = kron_solve(self.M_i, d)  
        return u_m_hat
    
    def project_initial_condition(self, u0):
        """
        Project the initial condition onto the trial basis at t=0.
        The trial basis is translated in time to t=-Dt for this projection.

        Args:
            u0 (see manufacturedsolutions.py)

        Returns:
            torch.Tensor: Projected initial condition coefficients, shape (B, N_0, ..., N_{d-1}).
        """
        self.phi.bases[0].Dx = -self.hparams['Dt'] # Translate basis in time to t=-Dt
        d = assemble_L(self.phi, u0, self.W_Gamma_t0, self.Xi_Gamma_t0, derivative_orders=(0,0,0)).flatten(start_dim=1)
        self.phi.bases[0].Dx = 0 # Translate basis in time back to t=0
        M_rr = kron(self.M_rr_i)
        M_rr_inv = torch.linalg.pinv(M_rr)
        u0_m = opt_einsum.contract('Nnm,Nm->Nn', M_rr_inv, d)
        return u0_m

    
    def assemble_F(self, theta):
        """
        Assemble the operator F based on the model type.

        Args:
            theta (ManufacturedSolution)

        Returns:
            torch.Tensor: Assembled operator F.
        """
        if self.hparams['project_materialparameters']==True:
            theta.u_m = self.project_input_function(theta)
            theta.basis = self.psi
        if  self.hparams['modeltype']=='data NGO':
            F = assemble_L(self.psi, theta, self.W_OmegaT, self.Xi_OmegaT, derivative_orders=(0,0,0))
        if  self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM' or self.hparams['modeltype']=='projection':
            #Steady diffusion part
            # Volume term - derivative in x direction
            T_i = assemble_T(theta, self.psi, self.phi, self.W_OmegaT, self.Xi_OmegaT, derivative_orders=((0,0,0),(0,1,0),(0,1,0)))
            F = kron_vecT(theta.u_m, T_i)
            # Volume term - derivative in y direction
            T_i = assemble_T(theta, self.psi, self.phi, self.W_OmegaT, self.Xi_OmegaT, derivative_orders=((0,0,0),(0,0,1),(0,0,1)))
            F += kron_vecT(theta.u_m, T_i)
            # Boundary term at x0 (coming from Neumann complement)
            T_i = assemble_T(theta, self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0,0),(0,0,0),(0,1,0)))
            F += kron_vecT(theta.u_m, T_i)
            # Boundary term at xL (coming from Neumann complement)
            T_i = assemble_T(theta, self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0,0),(0,0,0),(0,1,0)))
            F -= kron_vecT(theta.u_m, T_i)
            # Boundary term at x0 (coming from Dirichlet complement)
            T_i = assemble_T(theta, self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0,0),(0,1,0),(0,0,0)))
            F += kron_vecT(theta.u_m, T_i)
            # Boundary term at xL (coming from Dirichlet complement)
            T_i = assemble_T(theta, self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0,0),(0,1,0),(0,0,0)))
            F -= kron_vecT(theta.u_m, T_i)
            if self.hparams.get('gamma_stabilization',0)!=0:
                # Stabilization term at x0
                T_i = assemble_T(theta, self.psi, self.phi, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=((0,0,0),(0,0,0),(0,0,0)))
                F += self.hparams['gamma_stabilization']*kron_vecT(theta.u_m, T_i)
                # Stabilization term at xL
                T_i = assemble_T(theta, self.psi, self.phi, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=((0,0,0),(0,0,0),(0,0,0)))
                F += self.hparams['gamma_stabilization']*kron_vecT(theta.u_m, T_i)
            # Contract with theta
            # Time derivative term
            B_i  = kron(assemble_B(self.psi, self.phi, self.W_OmegaT, self.Xi_OmegaT, derivative_orders=((1,0,0),(0,0,0))))
            F -= B_i
            # Boundary term at t=T
            B_i= kron(assemble_B(self.psi, self.phi, self.W_Gamma_tT, self.Xi_Gamma_tT, derivative_orders=((0,0,0),(0,0,0))))
            F += B_i
        return F

    def assemble_d(self, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        """
        Compute the right-hand side vector d.

        Args:
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u0 (see manufacturedsolutions.py)

        Returns:
            torch.Tensor: Right-hand side vector d, shape (B, N).
        """
        #Steady diffusion part
        d = assemble_L(self.psi, f, self.W_OmegaT, self.Xi_OmegaT, derivative_orders=(0,0,0))
        d += assemble_L(self.psi, eta_y0, self.W_Gamma_y0, self.Xi_Gamma_y0, derivative_orders=(0,0,0))
        d += assemble_L(self.psi, eta_yL, self.W_Gamma_yL, self.Xi_Gamma_yL, derivative_orders=(0,0,0))
        d += assemble_L(self.psi, g_x0, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=(0,1,0)) # addition because of minus sign of boundary normal sign at x0
        d -= assemble_L(self.psi, g_xL, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=(0,1,0))
        if self.hparams.get('gamma_stabilization',0)!=0:
            d += self.hparams['gamma_stabilization']*assemble_L(self.psi, g_x0, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=(0,0,0))
            d += self.hparams['gamma_stabilization']*assemble_L(self.psi, g_xL, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=(0,0,0))
        # initial condition term
        d += assemble_L(self.psi, u0, self.W_Gamma_t0, self.Xi_Gamma_t0, derivative_orders=(0,0,0))
        # Flatten d
        d = d.flatten(start_dim=1)
        return d

    def assemble_d_x(self, f, eta_y0, eta_yL, g_x0, g_xL):
        """
        Compute the right-hand side vector d without the initial condition term.

        Args:
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)

        Returns:
            np.ndarray: Right-hand side vector d_x, shape (B, N).
        """
        #Steady diffusion part
        d = assemble_L(self.psi, f, self.W_OmegaT, self.Xi_OmegaT, derivative_orders=(0,0,0))
        d += assemble_L(self.psi, eta_y0, self.W_Gamma_y0, self.Xi_Gamma_y0, derivative_orders=(0,0,0))
        d += assemble_L(self.psi, eta_yL, self.W_Gamma_yL, self.Xi_Gamma_yL, derivative_orders=(0,0,0))
        d += assemble_L(self.psi, g_x0, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=(0,1,0)) # addition because of minus sign of boundary normal sign at x0
        d -= assemble_L(self.psi, g_xL, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=(0,1,0))
        if self.hparams.get('gamma_stabilization',0)!=0:
            d += self.hparams['gamma_stabilization']*assemble_L(self.psi, g_x0, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=(0,0,0))
            d += self.hparams['gamma_stabilization']*assemble_L(self.psi, g_xL, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=(0,0,0))
        # Flatten d
        d = d.flatten(start_dim=1)
        return d

    def assemble_Neumann_preconditioner(self):
        """
        Compute the matrices F_0 and A_0 for the Neumann series preconditioner.

        Returns:
            None
        """ 
        theta = UnitFunction(d=self.hparams['d'], device=self.device, dtype=self.dtype)
        self.F_0 = self.assemble_F(theta)/self.hparams['Dt']
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM' or self.hparams['modeltype']=='projection':
            self.A_0 = torch.linalg.inv(self.F_0)

    def assemble_M_i(self):
        """
        Compute the mass matrix M_i.

        Returns:
            M_i (tuple of torch.Tensor): Mass matrix M_i per dimension.
        """ 
        M_i = assemble_B(self.phi, self.phi, self.W_OmegaT, self.Xi_OmegaT, derivative_orders=((0,0,0),(0,0,0)))
        return M_i

    def assemble_M_rr_i(self):
        """
        Compute the mass matrix M_rr_i, corresponding to evaluation of the bases at the right end of the time element t=T.

        Returns:
            M_rr_i (tuple of torch.Tensor): Mass matrix M_rr_i per dimension.
        """ 
        M_rr_i = assemble_B(self.psi, self.phi, self.W_Gamma_tT, self.Xi_Gamma_tT, derivative_orders=((0,0,0),(0,0,0)))
        return M_rr_i
    
    def assemble_M_lr_i(self):
        """
        Compute the mass matrix M_lr_i, corresponding to evaluation of the bases at the 
        left (t=0) and right (t=T) ends of the time element.

        Returns:
            M_lr_i (tuple of torch.Tensor): Mass matrix M_lr_i per dimension.
        """ 
        self.phi.bases[0].Dx = -self.hparams['Dt'] # Translate basis in time to t=-Dt
        M_lr_i = assemble_B(self.psi, self.phi, self.W_Gamma_t0, self.Xi_Gamma_t0, derivative_orders=((0,0,0),(0,0,0)))
        self.phi.bases[0].Dx = 0 # Translate basis in time back to t=0
        return M_lr_i

    #Conserved quantity
    def compute_C(self, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        """
        Compute the conserved quantity C.

        Args:
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u0 (see manufacturedsolutions.py)

        Returns:
            torch.Tensor: Conserved quantity C, shape (B,).
        """
        C = integrate(self.W_OmegaT, f.forward(self.Xi_OmegaT))
        C += integrate(self.W_Gamma_y0, eta_y0.forward(self.Xi_Gamma_y0))
        C += integrate(self.W_Gamma_yL, eta_yL.forward(self.Xi_Gamma_yL))
        C += integrate(self.W_Gamma_t0, u0.forward(self.Xi_Gamma_t0))
        if self.hparams.get('stabilization_in_massconservation',False)==True:
            C += self.hparams['gamma_stabilization']*integrate(self.W_Gamma_x0, g_x0.forward(self.Xi_Gamma_x0))
            C += self.hparams['gamma_stabilization']*integrate(self.W_Gamma_xL, g_xL.forward(self.Xi_Gamma_xL))
        return C
   
    def compute_C_m(self, theta):
        """
        Compute the conserved quantity per degree of freedom C_m.

        Args:
            theta (ManufacturedSolution)

        Returns:
            torch.Tensor: Conserved quantity per degree of freedom C_m, shape (B, N).
        """
        C_m = assemble_L(self.phi, theta, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=(0,1,0))
        C_m += -assemble_L(self.phi, theta, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=(0,1,0))
        C_m += self.phi.integrals(self.W_Gamma_tT, self.Xi_Gamma_tT)
        if self.hparams.get('stabilization_in_massconservation',False)==True:
            C_m += self.hparams['gamma_stabilization']*assemble_L(self.phi, theta, self.W_Gamma_x0, self.Xi_Gamma_x0, derivative_orders=(0,0,0))
            C_m += self.hparams['gamma_stabilization']*assemble_L(self.phi, theta, self.W_Gamma_xL, self.Xi_Gamma_xL, derivative_orders=(0,0,0))
        C_m = C_m.flatten(start_dim=1)
        return C_m
        
    def NN_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        """
        Forward pass for the NN model (input functions -> solution).
        Args:
            theta_q (torch.Tensor): Discretized input function theta, shape (B, Q_0, Q_1, Q_2).
            f_q (torch.Tensor): Discretized input function f, shape (B, Q_0, Q_1, Q_2).
            eta_y0_q (torch.Tensor): Discretized input function eta_y0, shape (B, Q_0, Q_1, 1).
            eta_yL_q (torch.Tensor): Discretized input function eta_yL, shape (B, Q_0, Q_1, 1).
            g_x0_q (torch.Tensor): Discretized input function g_x0, shape (B, Q_0, 1, Q_2).
            g_xL_q (torch.Tensor): Discretized input function g_xL, shape (B, Q_0, 1, Q_2).
            u0_t0_q (torch.Tensor): Discretized initial condition function u0 at t=0, shape (B, 1, Q_1, Q_2).

        Returns:
            torch.Tensor: Predicted solution u_q_hat, shape (B, Q_0, Q_1, Q_2).
        """     
        eta_q = torch.zeros((eta_y0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        eta_q[:,:,:,0] = eta_y0_q[:,:,:,0]
        eta_q[:,:,:,-1] = eta_yL_q[:,:,:,0]
        g_q = torch.zeros((g_x0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        g_q[:,:,0,:] = g_x0_q[:,:,0,:]
        g_q[:,:,-1,:] = g_xL_q[:,:,0,:]
        u0_t0 = torch.zeros((u0_t0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        u0_t0[:,0,:,:] = u0_t0_q[:,0,:,:]
        u0_t0_q = u0_t0
        inputfuncs = torch.stack((theta_q,f_q,eta_q,g_q,u0_t0_q), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta_q.flatten(-3,-1),f_q.flatten(-3,-1),eta_y0_q,eta_yL_q,g_x0_q,g_xL_q,u0_t0_q), dim=1)
        u_q_hat = self.systemnet.forward(inputfuncs)
        return u_q_hat
    
    def NN_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        """
        Simulation forward pass for the NN model (input functions -> solution).

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u0 (see manufacturedsolutions.py)
            u (ManufacturedSolution)

        Returns:
            np.ndarray: Predicted solution u_q_hat, shape (B, Q_0, Q_1, Q_2).
        """
        theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
        # If not first timestep, use previous solution as initial condition
        if self.timestep>0:
            u0_t0_q = torch.tensor(self.u_q_hat, dtype=self.dtype, device=self.device)
            u0_t0_q = u0_t0_q.reshape(((u0_t0_q.shape[0],)+self.hparams['Q']))
            u0_t0_q = u0_t0_q[:,-1]
        # Take batches of 100 samples to avoid GPU memory overflow for FNO
        if self.hparams['systemnet']==FNO:
            bs = 100
            if theta_q.shape[0]>bs:
                u_q_hat = np.zeros((theta_q.shape[0],)+(np.prod(self.hparams['Q_L']),))
                for i in range(int(theta_q.shape[0]/bs)):
                    u_q_hat[bs*i:bs*(i+1)] = self.NN_forward(theta_q[bs*i:bs*(i+1)], f_q[bs*i:bs*(i+1)], eta_y0_q[bs*i:bs*(i+1)], eta_yL_q[bs*i:bs*(i+1)], g_x0_q[bs*i:bs*(i+1)], g_xL_q[bs*i:bs*(i+1)], u0_t0_q[bs*i:bs*(i+1)]).detach().cpu().numpy()
            else:
                u_q_hat = self.NN_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q).detach().cpu().numpy()
        # Else for other system nets, process all samples at once
        else:
            u_q_hat = self.NN_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q).detach().cpu().numpy()
        # Store solution for next timestep
        self.u_q_hat = u_q_hat
        return u_q_hat
    
    def DeepONet_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        """
        Forward pass for the DeepONet model.

        Args:
            theta_q (torch.Tensor): Discretized input function theta, shape (B, Q_0, Q_1, Q_2).
            f_q (torch.Tensor): Discretized input function f, shape (B, Q_0, Q_1, Q_2).
            eta_y0_q (torch.Tensor): Discretized input function eta_y0, shape (B, Q_0, Q_1, 1).
            eta_yL_q (torch.Tensor): Discretized input function eta_yL, shape (B, Q_0, Q_1, 1).
            g_x0_q (torch.Tensor): Discretized input function g_x0, shape (B, Q_0, 1, Q_2).
            g_xL_q (torch.Tensor): Discretized input function g_xL, shape (B, Q_0, 1, Q_2).
            u0_t0_q (torch.Tensor): Discretized initial condition function u0 at t=0, shape (B, 1, Q_1, Q_2).

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """      
        eta_q = torch.zeros((eta_y0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        eta_q[:,:,:,0] = eta_y0_q[:,:,:,0]
        eta_q[:,:,:,-1] = eta_yL_q[:,:,:,0]
        g_q = torch.zeros((g_x0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        g_q[:,:,0,:] = g_x0_q[:,:,0,:]
        g_q[:,:,-1,:] = g_xL_q[:,:,0,:]
        u0_t0 = torch.zeros((u0_t0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        u0_t0[:,0,:,:] = u0_t0_q[:,0,:,:]
        u0_t0_q = u0_t0
        inputfuncs = torch.stack((theta_q,f_q,eta_q,g_q,u0_t0_q), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta_q.flatten(-3,-1),f_q.flatten(-3,-1),eta_y0_q,eta_yL_q,g_x0_q,g_xL_q,u0_t0_q), dim=1)
        u_m_hat = self.systemnet.forward(inputfuncs)
        return u_m_hat
    
    def DeepONet_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        """
        Simulation forward pass for the DeepONet model.

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u0 (see manufacturedsolutions.py)
            u (ManufacturedSolution)            
        
        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
            """
        theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
        # If not first timestep, use previous solution as initial condition
        if self.timestep>0:
            u0_t0_q = self.phi.forward(self.u_m_hat, self.Xi_Gamma_tT)
        u_m_hat = self.DeepONet_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q)
        # Store solution for next timestep
        self.u_m_hat = u_m_hat
        return u_m_hat
    
    def VarMiON_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        """
        Forward pass for the VarMiON model.

        Args:
            theta_q (torch.Tensor): Discretized input function theta, shape (N_samples, Q_0, Q_1, Q_2).
            f_q (torch.Tensor): Discretized input function f, shape (N_samples, Q_0, Q_1, Q_2).
            eta_y0_q (torch.Tensor): Discretized input function eta_y0, shape (N_samples, Q_0, Q_1, 1).
            eta_yL_q (torch.Tensor): Discretized input function eta_yL, shape (N_samples, Q_0, Q_1, 1).
            g_x0_q (torch.Tensor): Discretized input function g_x0, shape (N_samples, Q_0, 1, Q_2).
            g_xL_q (torch.Tensor): Discretized input function g_xL, shape (N_samples, Q_0, 1, Q_2).
            u0_t0_q (torch.Tensor): Discretized initial condition function u0 at t=0, shape (N_samples, 1, Q_1, Q_2).

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
        u0_t0_q = u0_t0_q.flatten(start_dim=1)
        A_hat = self.systemnet.forward(theta_q)
        if self.hparams.get('Mnorm_scaling',False)==True:
            A_hat = M_norm_scaling(self, A_hat)
        d_hat = self.LBranch_f.forward(f_q) + self.LBranch_eta.forward(eta_q) + self.LBranch_g.forward(g_q) + self.LBranch_u0.forward(u0_t0_q)
        u_m_hat = opt_einsum.contract('nij,nj->ni', A_hat, d_hat)
        return u_m_hat
    
    def VarMiON_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        """
        Simulation forward pass for the VarMiON model.

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u0 (see manufacturedsolutions.py)
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """ 
        theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
        # If not first timestep, use previous solution as initial condition
        if self.timestep>0:
            u0_t0_q = self.phi.forward(self.u_m_hat, self.Xi_Gamma_tT)
        u_m_hat = self.VarMiON_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q)
        # Store solution for next timestep
        self.u_m_hat = u_m_hat
        return u_m_hat
    
    def NGO_forward(self, F, d, d_x, u0_m, C, C_m, theta_bar):
        """
        Forward pass for the NGO model.

        Args:
            F (torch.Tensor): Input tensor F, shape (B, N, N).
            d (torch.Tensor): Input tensor d, shape (B, N).
            d_x (torch.Tensor): Input tensor d_x, shape (B, N).
            u0_m (torch.Tensor): Input tensor u0_m, shape (B, N).
            C (torch.Tensor): Input tensor C, shape (B, N, N).
            C_m (torch.Tensor): Input tensor C_m, shape (B, N, N).
            theta_bar (torch.Tensor): Input tensor theta_bar, shape (B, N).

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """
        if self.hparams.get('scaling_equivariance',False)==True:
            F = F/theta_bar[:,None,None,None]
        if self.hparams.get('Neumannseries', False)==False:
            A_hat = self.systemnet.forward(F)*self.hparams['Dt']
        if self.hparams.get('Neumannseries', False)==True:
            A_hat = Neumann_preconditioner_forward(self, F)
        if self.hparams.get('scaling_equivariance',False)==True:
            A_hat = A_hat/theta_bar[:,None,None]
        if self.hparams.get('Mnorm_scaling',False)==True:
            A_hat = M_norm_scaling(self, A_hat)
        u_m_hat = opt_einsum.contract('nij,nj->ni', A_hat, d)
        if self.hparams['massconservation']==True:
            u_m_hat = conserve_mass(C_m, C, u_m_hat)
        if self.hparams['dissipation']==True and self.hparams['massconservation']==False:
            u_m_hat = conserve_energy(self, F, d_x, u0_m, u_m_hat)
        if self.hparams['massconservation']==True and self.hparams['dissipation']==True:
            u_m_hat = conserve_mass_and_energy(self, d_x, u0_m, C_m, C, u_m_hat)
        return u_m_hat
    
    def NGO_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        """
        Simulation forward pass for the NGO model.

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u0 (see manufacturedsolutions.py)
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """
        # # If not first timestep, use previous solution as initial condition
        if self.timestep==0:
            u0_m = self.project_initial_condition(u0)
        if self.timestep>0:
            # Translate basis in time to t=-Dt
            self.phi.bases[0].Dx += -self.hparams['Dt']
            # Define new initial condition function at t=0 based on previous solution at t=Dt
            u0.forward = lambda x: self.phi.forward(self.u_m_hat, (x[0]+self.hparams['Dt'], x[1], x[2]))
            # Translate basis in time back to t=0
            self.phi.bases[0].Dx += self.hparams['Dt']
            # Use projected previous solution as initial condition
            u0_m = self.u_m_hat
        # Assemble system
        F = self.assemble_F(theta)/self.hparams['Dt']
        d = self.assemble_d(f, eta_y0, eta_yL, g_x0, g_xL, u0)/self.hparams['Dt']
        d_x = self.assemble_d_x(f, eta_y0, eta_yL, g_x0, g_xL)/self.hparams['Dt']
        C = self.compute_C(f, eta_y0, eta_yL, g_x0, g_xL, u0)/self.hparams['Dt']
        C_m = self.compute_C_m(theta)/self.hparams['Dt']
        theta_bar = theta.basis.integrate(theta.u_m, self.W_OmegaT, self.Xi_OmegaT)/self.hparams['Dt']
        # Compute solution
        u_m_hat = self.NGO_forward(F, d, d_x, u0_m, C, C_m, theta_bar)
        # Store solution for next timestep
        self.u_m_hat = u_m_hat
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
        if self.hparams['Neumannseries']==False:
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
    
    def FEM_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        """
        Simulation forward pass for the FEM model.

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u0 (see manufacturedsolutions.py)
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """
        # If not first timestep, use previous solution as initial condition
        if self.timestep>0:
            # Translate basis in time to t=-Dt
            self.phi.bases[0].Dx += -self.hparams['Dt']
            # Define new initial condition function at t=0 based on previous solution at t=Dt
            u0.forward = lambda x: self.phi.forward(self.u_m_hat, (x[0]+self.hparams['Dt'], x[1], x[2]))
            # Translate basis in time back to t=0
            self.phi.bases[0].Dx += self.hparams['Dt']
        # Assemble system
        F = self.assemble_F(theta)
        d = self.assemble_d(f, eta_y0, eta_yL, g_x0, g_xL, u0)
        # Compute solution
        u_m_hat = self.FEM_forward(F, d)
        # Store solution for next timestep
        self.u_m_hat = u_m_hat
        return u_m_hat
    
    def projection_forward(self, u):
        """
        L2 projection of the solution onto the trial basis functions.

        Args:
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Projected solution coefficients u_m_hat, shape (N_samples, N).
        """
        u_m_hat = self.project_output_function(u)
        return u_m_hat
    
    def projection_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        """
        Simulation forward pass for the projection model.

        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u0 (see manufacturedsolutions.py)
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """
        u_m_hat = self.projection_forward(u)
        self.u_m_hat = u_m_hat
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
                u_q_hat = self.phi.forward(u_m, self.Xi_OmegaT_L)
                return u_q_hat

    def simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, X, u):
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
        u_m_hat = self.simforwardfunction(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u)
        u_q_hat = u_m_hat if self.hparams['modeltype']=='NN' else self.phi.forward(u_m_hat, X)
        self.time_step()
        return u_q_hat

    def simforward_coefficients(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        """
        Simulation forward pass that returns the solution coefficients for the model,
        set in the function self.init_modeltype(), based on the selected hparams['modeltype'].
        Args:
            theta (ManufacturedSolution)
            f (see manufacturedsolutions.py)
            eta_y0 (see manufacturedsolutions.py)
            eta_yL (see manufacturedsolutions.py)
            g_x0 (see manufacturedsolutions.py)
            g_xL (see manufacturedsolutions.py)
            u0 (see manufacturedsolutions.py)
            u (ManufacturedSolution)

        Returns:
            torch.Tensor: Predicted solution coefficients u_m_hat, shape (B, N).
        """
        u_m_hat = self.simforwardfunction(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u)
        return u_m_hat

    def compute_quadrature(self):
        """
        Compute the quadrature points and weights.

        Returns:
            None
        """
        # Base points
        self.zero = torch.zeros(size=(1,), dtype=self.dtype, device=self.device)
        self.one = torch.ones(size=(1,), dtype=self.dtype, device=self.device)
        self.t_0 = torch.zeros(size=(1,), dtype=self.dtype, device=self.device)
        self.t_1 = torch.ones(size=(1,), dtype=self.dtype, device=self.device)
        if self.hparams['quadrature']=='uniform':
            self.w_t, self.xi_t = UniformQuadrature(Q=self.hparams['Q'][0], dtype=self.dtype, device=self.device).compute_quadrature()
            self.w_x, self.xi_x = UniformQuadrature(Q=self.hparams['Q'][1], dtype=self.dtype, device=self.device).compute_quadrature()
            self.w_y, self.xi_y = UniformQuadrature(Q=self.hparams['Q'][2], dtype=self.dtype, device=self.device).compute_quadrature()
        if self.hparams['quadrature']=='Gauss-Legendre':
            self.w_t, self.xi_t = GaussLegendreQuadrature(Q=self.hparams['Q'][0], n_elements=self.hparams['n_elements'][0], dtype=self.dtype, device=self.device).compute_quadrature()
            self.w_x, self.xi_x = GaussLegendreQuadrature(Q=self.hparams['Q'][1], n_elements=self.hparams['n_elements'][1], dtype=self.dtype, device=self.device).compute_quadrature()
            self.w_y, self.xi_y = GaussLegendreQuadrature(Q=self.hparams['Q'][2], n_elements=self.hparams['n_elements'][2], dtype=self.dtype, device=self.device).compute_quadrature()
        # Scale quadrature in time by Dt
        self.w_t, self.xi_t = self.hparams['Dt']*self.w_t, self.hparams['Dt']*self.xi_t
        self.t_0 = self.hparams['Dt']*self.t_0
        self.t_1 = self.hparams['Dt']*self.t_1
        self.xi_t_0 = self.xi_t.clone()
        # Domain quadrature
        self.W_OmegaT = self.w_t, self.w_x, self.w_y
        self.Xi_OmegaT = self.xi_t, self.xi_x, self.xi_y
        # Boundary x0 quadrature
        self.W_Gamma_x0 = self.w_t, self.one, self.w_y
        self.Xi_Gamma_x0 = self.xi_t, self.zero, self.xi_y
        # Boundary xL quadrature
        self.W_Gamma_xL = self.w_t, self.one, self.w_y
        self.Xi_Gamma_xL = self.xi_t, self.one, self.xi_y
        # Boundary y0 quadrature
        self.W_Gamma_y0 = self.w_t, self.w_x, self.one
        self.Xi_Gamma_y0 = self.xi_t, self.xi_x, self.zero
        # Boundary yL quadrature
        self.W_Gamma_yL = self.w_t, self.w_x, self.one
        self.Xi_Gamma_yL = self.xi_t, self.xi_x, self.one
        # Boundary t0 quadrature
        self.W_Gamma_t0 = self.one, self.w_x, self.w_y
        self.Xi_Gamma_t0 = self.t_0, self.xi_x, self.xi_y
        # Boundary tT quadrature
        self.W_Gamma_tT = self.one, self.w_x, self.w_y
        self.Xi_Gamma_tT = self.t_1, self.xi_x, self.xi_y

        #Loss quadrature
        if self.hparams['quadrature_L']=='uniform':
            self.w_t_L, self.xi_t_L = UniformQuadrature(Q=self.hparams['Q_L'][0], dtype=self.dtype, device=self.device).compute_quadrature()
            self.w_x_L, self.xi_x_L = UniformQuadrature(Q=self.hparams['Q_L'][1], dtype=self.dtype, device=self.device).compute_quadrature()
            self.w_y_L, self.xi_y_L = UniformQuadrature(Q=self.hparams['Q_L'][2], dtype=self.dtype, device=self.device).compute_quadrature()
        if self.hparams['quadrature_L']=='Gauss-Legendre':
            self.w_t_L, self.xi_t_L = GaussLegendreQuadrature(Q=self.hparams['Q_L'][0], n_elements=self.hparams['n_elements_L'][0], dtype=self.dtype, device=self.device).compute_quadrature()
            self.w_x_L, self.xi_x_L = GaussLegendreQuadrature(Q=self.hparams['Q_L'][1], n_elements=self.hparams['n_elements_L'][1], dtype=self.dtype, device=self.device).compute_quadrature()
            self.w_y_L, self.xi_y_L = GaussLegendreQuadrature(Q=self.hparams['Q_L'][2], n_elements=self.hparams['n_elements_L'][2], dtype=self.dtype, device=self.device).compute_quadrature()
        # Scale loss quadrature in time by Dt
        self.w_t_L, self.xi_t_L = self.hparams['Dt']*self.w_t_L, self.hparams['Dt']*self.xi_t_L
        # Loss domain quadrature
        self.W_OmegaT_L = self.w_t_L, self.w_x_L, self.w_y_L
        self.Xi_OmegaT_L = self.xi_t_L, self.xi_x_L, self.xi_y_L

    def time_step(self):
        """
        Translate the quadrature points and bases in the time dimension by the time step size Dt.
        """
        # Update time
        self.timestep += int(1)
        # Update time element boundaries
        self.t_0 = self.timestep*self.hparams['Dt']*self.one
        self.t_1 = self.t_0 + self.hparams['Dt']
        # Update time quadrature points
        self.xi_t =  self.xi_t_0 + self.t_0
        # Shift bases forward in time
        self.psi.bases[0].Dx = self.t_0
        self.phi.bases[0].Dx = self.t_0

        # Update quadrature points
        # Domain quadrature
        self.W_OmegaT = self.w_t, self.w_x, self.w_y
        self.Xi_OmegaT = self.xi_t, self.xi_x, self.xi_y
        # Boundary x0 quadrature
        self.W_Gamma_x0 = self.w_t, self.one, self.w_y
        self.Xi_Gamma_x0 = self.xi_t, self.zero, self.xi_y
        # Boundary xL quadrature
        self.W_Gamma_xL = self.w_t, self.one, self.w_y
        self.Xi_Gamma_xL = self.xi_t, self.one, self.xi_y
        # Boundary y0 quadrature
        self.W_Gamma_y0 = self.w_t, self.w_x, self.one
        self.Xi_Gamma_y0 = self.xi_t, self.xi_x, self.zero
        # Boundary yL quadrature
        self.W_Gamma_yL = self.w_t, self.w_x, self.one
        self.Xi_Gamma_yL = self.xi_t, self.xi_x, self.one
        # Boundary t0 quadrature
        self.W_Gamma_t0 = self.one, self.w_x, self.w_y
        self.Xi_Gamma_t0 = self.t_0, self.xi_x, self.xi_y
        # Boundary tT quadrature
        self.W_Gamma_tT = self.one, self.w_x, self.w_y
        self.Xi_Gamma_tT = self.t_1, self.xi_x, self.xi_y

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
                u_hat = u_hat.reshape((u_hat.shape[0],)+self.hparams['N_i'])
                loss += self.hparams['solution_loss'](self.M_i, u_hat, u)
            if self.hparams['output_coefficients']==False:
                loss += self.hparams['solution_loss'](self.W_OmegaT_L, u_hat, u)
        if self.hparams.get('matrix_loss',None)!=None:
            F = inputs[0]
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
            if self.hparams['output_coefficients']==True:
                u_hat = u_hat.reshape((u_hat.shape[0],)+self.hparams['N_i'])
                loss += self.hparams['solution_loss'](self.M_i, u_hat, u)
            if self.hparams['output_coefficients']==False:
                loss += self.hparams['solution_loss'](self.W_OmegaT_L, u_hat, u)
        if self.hparams.get('matrix_loss',None)!=None:
            F = inputs[0]
            if self.hparams.get('Neumannseries',False)==False:
                A_hat = self.systemnet.forward(F)
            if self.hparams.get('Neumannseries', False)==True:
                A_hat = Neumann_preconditioner_forward(self, F)
            loss += self.hparams['matrix_loss'](torch.matmul(F, torch.matmul(A_hat, F)), F)
            print(loss)
        if self.hparams['output_coefficients']==True:
            metric = self.hparams['metric'](self.M_i, u_hat, u)
        if self.hparams['output_coefficients']==False:
            metric = self.hparams['metric'](self.W_OmegaT_L, u_hat, u)
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
        # Limit number of threads for linear algebra operations
        torch.set_num_threads(2)
        # Initialize best metric value
        self.metric_best = 100
        # Move systemnet to device
        self.systemnet.device = self.device

    def on_validation_epoch_end(self):
        """
        Perform actions at the end of a validation epoch (hook called by pytorch lightning under the hood).

        Returns:
            None
        """
        # If the metric improves, update the dominant eigenvector estimation
        if self.metric<self.metric_best:
            self.metric_best = self.metric
            if self.hparams.get('Mnorm_scaling',False)==True:
                # System net output for F_0 (for average theta)
                A_hat = self.systemnet.forward(self.F_0)
                # Compute update matrix
                B = A_hat@self.M_lr
                # Extract relevant block
                B = B[:,int(A_hat.shape[1]/2):,int(A_hat.shape[2]/2):]
                # Compute transformed matrix for 2-norm estimation
                C = self.M_rr22_05 @ B @ self.M_rr22_05_inv
                # Update 2-norm and dominant eigenvector estimate
                Mnorm_0, self.x_0 = two_norm_power_iteration(B=C, norm2_0=self.Mnorm_0, x_0=self.x_0, tol=self.hparams['tol'], max_iterations=self.hparams['max_iterations'], dtype=self.dtype, device=self.device)
                # Save updated dominant eigenvector
                torch.save(self.x_0.cpu(), self.hparams['logdir']+'/x_0.pt')
        print("Validation metric: " +str(self.metric))
