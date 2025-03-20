import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
import numpy as np
import opt_einsum

import sys
sys.path.insert(0, '../../ml')
from systemnets import MLP, CNN, FNO, LBranchNet
from basisfunctions import *
from quadrature import *
from customlayers import *
from customlosses import *
from modelloader import *

class NeuralOperator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.to(self.hparams['device'])
        self.to(self.hparams['dtype'])
        #compute_quadrature
        self.compute_quadrature()
        #Model type
        self.init_modeltype()
        #System net
        self.hparams['N_w_real'] = sum(p.numel() for p in self.parameters())
        self.systemnet = self.hparams['systemnet'](self.hparams).to(self.device)
        #Bases
        if self.hparams.get('POD',False)==False:
            self.basis_test = TensorizedBasis(self.hparams['test_bases']) 
            self.basis_trial = TensorizedBasis(self.hparams['trial_bases'])
        if self.hparams.get('POD',False)==True:
            self.basis_test = BSplineInterpolatedPOD2D(N_samples=self.hparams['N_samples_train'], variables=hparams['variables'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'], w=self.w_Omega, xi=self.xi_Omega, N=self.hparams['N'], device=self.device)
            self.basis_trial = BSplineInterpolatedPOD2D(N_samples=self.hparams['N_samples_train'], variables=hparams['variables'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'], w=self.w_Omega, xi=self.xi_Omega, N=self.hparams['N'], device=self.device)
        #Relevant matrices
        if self.hparams['Neumannseries']==True:
            self.compute_F_0_A_0()
        #Identity
        self.Identity = torch.eye(self.hparams['N'], dtype=self.dtype, device=self.device)

    def init_modeltype(self):
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

    def discretize_input_functions(self, theta, f, eta_y0, eta_yL, g_x0, gr):
        theta_d = discretize_functions(theta, self.xi_Omega)
        theta_x0_d = discretize_functions(theta, self.xi_Gamma_x0)
        theta_xL_d = discretize_functions(theta, self.xi_Gamma_xL)
        f_d = discretize_functions(f, self.xi_Omega)
        eta_y0_d = discretize_functions(eta_y0, self.xi_Gamma_y0)
        eta_yL_d = discretize_functions(eta_yL, self.xi_Gamma_yL)
        g_x0_d = discretize_functions(g_x0, self.xi_Gamma_x0)
        g_xL_d = discretize_functions(gr, self.xi_Gamma_xL)
        return theta_d, theta_x0_d, theta_xL_d, f_d, eta_y0_d, eta_yL_d, g_x0_d, g_xL_d
    
    def discretize_output_function(self, u):
        u_d = discretize_functions(u, self.xi_Omega_L)
        return u_d

    def compute_F(self, theta, theta_x0, theta_xL):
        if  self.hparams['modeltype']=='data NGO':
            basis_test = self.basis_test.forward(self.xi_Omega)
            F = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, theta)
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
            F = opt_einsum.contract('q,Nq,qmx,qnx->Nmn', self.w_Omega, theta, gradbasis_test, gradbasis_trial)
            F += -opt_einsum.contract('q,qm,x,Nq,qnx->Nmn', self.w_Gamma_x0, basis_test_x0, self.n_x0, theta_x0, gradbasis_trial_x0)
            F += -opt_einsum.contract('q,qm,x,Nq,qnx->Nmn', self.w_Gamma_xL, basis_test_xL, self.n_xL, theta_xL, gradbasis_trial_xL)
            F += -opt_einsum.contract('q,qn,x,Nq,qmx->Nmn', self.w_Gamma_x0, basis_trial_x0, self.n_x0, theta_x0, gradbasis_test_x0)
            F += -opt_einsum.contract('q,qn,x,Nq,qmx->Nmn', self.w_Gamma_xL, basis_trial_xL, self.n_xL, theta_xL, gradbasis_test_xL)
            if self.hparams.get('gamma_stabilization',0)!=0:
                F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_x0, basis_test_x0, theta_x0, basis_trial_x0)
                F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_xL, basis_test_xL, theta_xL, basis_trial_xL)
        return F
    
    def compute_F_0_A_0(self):
        theta = np.ones(self.w_Omega.shape)
        theta_x0 = np.ones(self.w_Gamma_x0.shape)
        theta_xL = np.ones(self.w_Gamma_xL.shape)
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
        F_0 = opt_einsum.contract('q,q,qmx,qnx->mn', self.w_Omega, theta, gradbasis_test, gradbasis_trial)
        F_0 += -opt_einsum.contract('q,qm,x,q,qnx->mn', self.w_Gamma_x0, basis_test_x0, self.n_x0, theta_x0, gradbasis_trial_x0)
        F_0 += -opt_einsum.contract('q,qm,x,q,qnx->mn', self.w_Gamma_xL, basis_test_xL, self.n_xL, theta_xL, gradbasis_trial_xL)
        F_0 += -opt_einsum.contract('q,qn,x,q,qmx->mn', self.w_Gamma_x0, basis_trial_x0, self.n_x0, theta_x0, gradbasis_test_x0)
        F_0 += -opt_einsum.contract('q,qn,x,q,qmx->mn', self.w_Gamma_xL, basis_trial_xL, self.n_xL, theta_xL, gradbasis_test_xL)
        if self.hparams.get('gamma_stabilization',0)!=0:
            F_0 += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,q,qn->mn', self.w_Gamma_x0, basis_test_x0, theta_x0, basis_trial_x0)
            F_0 += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,q,qn->mn', self.w_Gamma_xL, basis_test_xL, theta_xL, basis_trial_xL)
        self.F_0 = torch.tensor(F_0, dtype=self.dtype, device=self.device)
        self.A_0 = torch.linalg.inv(self.F_0)
    
    def compute_d(self, f, eta_y0, eta_yL, g_x0, gr):
        basis_test = self.basis_test.forward(self.xi_Omega)
        basis_test_b = self.basis_test.forward(self.xi_Gamma_y0)
        basis_test_t = self.basis_test.forward(self.xi_Gamma_yL)
        basis_test_l = self.basis_test.forward(self.xi_Gamma_x0)
        basis_test_r = self.basis_test.forward(self.xi_Gamma_xL)
        gradbasis_test_l = self.basis_test.grad(self.xi_Gamma_x0)
        gradbasis_test_r = self.basis_test.grad(self.xi_Gamma_xL)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, f)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_y0, basis_test_b, eta_y0)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_yL, basis_test_t, eta_yL)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_x0, self.n_x0, gradbasis_test_l, g_x0)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_xL, self.n_xL, gradbasis_test_r, gr)
        if self.hparams.get('gamma_stabilization',0)!=0:
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_x0, basis_test_l, g_x0)
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_xL, basis_test_r, gr)                 
        return d
    
    def NN_forward(self, theta, f, eta_y0, eta_yL, g_x0, gr):
        theta = theta.reshape((theta.shape[0],self.hparams['Q'],self.hparams['Q']))
        f = f.reshape((f.shape[0],self.hparams['Q'],self.hparams['Q']))
        eta = torch.zeros((eta_y0.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.dtype, device=self.device)
        eta[:,:,0] = eta_y0
        eta[:,:,-1] = eta_yL
        g = torch.zeros((g_x0.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.dtype, device=self.device)
        g[:,0,:] = g_x0
        g[:,-1,:] = gr
        inputfuncs = torch.stack((theta,f,eta,g), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta.flatten(-2,-1),f.flatten(-2,-1),eta_y0,eta_yL,g_x0,gr), dim=1)
        u_hat = self.systemnet.forward(inputfuncs).reshape((theta.shape[0],self.hparams['Q_L']**self.hparams['d']))
        return u_hat
    
    def NN_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        theta_q = torch.tensor(theta_q, dtype=self.dtype, device=self.device)
        f_q = torch.tensor(f_q, dtype=self.dtype, device=self.device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.dtype, device=self.device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.dtype, device=self.device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.dtype, device=self.device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.dtype, device=self.device)   
        u_q_hat = self.NN_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q).detach().cpu().numpy()
        return u_q_hat
    
    def DeepONet_forward(self, theta, f, eta_y0, eta_yL, g_x0, gr):
        theta = theta.reshape((theta.shape[0],self.hparams['Q'],self.hparams['Q']))
        f = f.reshape((f.shape[0],self.hparams['Q'],self.hparams['Q']))
        eta = torch.zeros((eta_y0.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.dtype, device=self.device)
        eta[:,:,0] = eta_y0
        eta[:,:,-1] = eta_yL
        g = torch.zeros((g_x0.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.dtype, device=self.device)
        g[:,0,:] = g_x0
        g[:,-1,:] = gr
        inputfuncs = torch.stack((theta,f,eta,g), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta.flatten(-2,-1),f.flatten(-2,-1),eta_y0,eta_yL,g_x0,gr), dim=1)
        u_n = self.systemnet.forward(inputfuncs).reshape((theta.shape[0],self.hparams['N']))
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat
    
    def DeepONet_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        theta_q = torch.tensor(theta_q, dtype=self.dtype, device=self.device)
        f_q = torch.tensor(f_q, dtype=self.dtype, device=self.device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.dtype, device=self.device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.dtype, device=self.device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.dtype, device=self.device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.dtype, device=self.device)   
        u_hat = self.DeepONet_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q).detach().cpu().numpy()
        return u_hat
    
    def VarMiON_forward(self, theta, f, eta_y0, eta_yL, g_x0, gr):
        eta = torch.zeros((eta_y0.shape[0],2*eta_y0.shape[1]), dtype=self.dtype, device=self.device)
        eta[:,:eta_y0.shape[1]] = eta_y0
        eta[:,eta_y0.shape[1]:] = eta_yL
        g = torch.zeros((g_x0.shape[0],2*g_x0.shape[1]), dtype=self.dtype, device=self.device)
        g[:,:g_x0.shape[1]] = g_x0
        g[:,g_x0.shape[1]:] = gr
        systemnet = self.systemnet.forward(theta)
        LBranch = self.LBranch_f.forward(f) + self.LBranch_eta.forward(eta) + self.LBranch_g.forward(g)
        u_n = opt_einsum.contract('nij,nj->ni', systemnet, LBranch)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat
    
    def VarMiON_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        theta_q = torch.tensor(theta_q, dtype=self.dtype, device=self.device)
        f_q = torch.tensor(f_q, dtype=self.dtype, device=self.device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.dtype, device=self.device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.dtype, device=self.device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.dtype, device=self.device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.dtype, device=self.device)   
        u_hat = self.VarMiON_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q).detach().cpu().numpy()
        return u_hat

    def NGO_forward(self, scaling, F, d):
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
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat
    
    def NGO_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        F = torch.tensor(self.compute_F(theta_q, theta_x0_q, theta_xL_q), dtype=self.dtype, device=self.device)
        d = torch.tensor(self.compute_d(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q), dtype=self.dtype, device=self.device)
        scaling = torch.tensor(np.sum(self.w_Omega[None,:]*theta_q, axis=-1), dtype=self.dtype, device=self.device)
        u_hat = self.NGO_forward(scaling, F, d).detach().cpu().numpy()
        return u_hat
    
    def FEM_forward(self, F, d):
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
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat
    
    def FEM_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        F = torch.tensor(self.compute_F(theta_q, theta_x0_q, theta_xL_q), dtype=self.dtype, device=self.device)
        d = torch.tensor(self.compute_d(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q), dtype=self.dtype, device=self.device)
        u_hat = self.FEM_forward(F, d).detach().cpu().numpy()
        return u_hat
    
    def projection_forward(self, u):
        u_q = discretize_functions(u, self.xi_Omega)
        basis_trial = self.basis_trial.forward(self.xi_Omega)
        u_w = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_trial, u_q)
        M = opt_einsum.contract('q,qm,qn->mn', self.w_Omega, basis_trial, basis_trial)
        M_inv = np.linalg.pinv(M)
        u_n = torch.tensor(opt_einsum.contract('mn,Nm->Nn', M_inv, u_w), dtype=self.dtype, device=self.device)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat
    
    def projection_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u):
        u_hat = self.projection_forward(u)
        return u_hat
    
    def forward(self, *args):
        u_hat = self.forwardfunction(*args)
        return u_hat
    
    def simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, x, u):
        self.psix = torch.tensor(self.basis_trial.forward(x), dtype=self.dtype, device=self.device)
        u_hat = self.simforwardfunction(theta, f, eta_y0, eta_yL, g_x0, g_xL, u)
        return u_hat
    
    def compute_quadrature(self):
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
        optimizer = self.hparams['optimizer'](self.parameters(), lr=self.hparams['learning_rate'])
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
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
        checkpoint['hparams'] = self.hparams
        
    def on_fit_start(self):
        torch.set_num_threads(2)
        #Basis evaluation at loss quadrature points
        self.psix = torch.tensor(self.basis_trial.forward(self.xi_Omega_L), dtype=self.dtype, device=self.device).to(self.device)
        self.w_Omega_L = torch.tensor(self.w_Omega_L).to(self.device)
        self.systemnet.device = self.device

    def on_validation_epoch_end(self):
        print(self.metric)
