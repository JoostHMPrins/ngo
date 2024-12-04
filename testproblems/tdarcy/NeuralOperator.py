import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
import numpy as np
import opt_einsum

import sys
sys.path.insert(0, '/home/prins/st8/prins/phd/gitlab/ngo-pde-gk/ml')
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
        self.used_device = self.hparams['used_device']
        #Model type
        self.init_modeltype()
        #Linear branches
        if self.hparams['modeltype']=='VarMiON':
            self.LBranch_f = LBranchNet(hparams, input_dim=self.hparams['Q']**self.hparams['d'], output_dim=self.hparams['N'])
            self.LBranch_eta = LBranchNet(hparams, input_dim=2*self.hparams['Q'], output_dim=self.hparams['N'])
            self.LBranch_g = LBranchNet(hparams, input_dim=2*self.hparams['Q'], output_dim=self.hparams['N'])
        self.hparams['N_w_real'] = sum(p.numel() for p in self.parameters())
        #System net
        self.systemnet = self.hparams['systemnet'](self.hparams)
        #Geometry and quadrature
        self.geometry()
        #Bases
        if self.hparams.get('POD',False)==False:
            self.basis_test = TensorizedBasis(self.hparams['test_bases']) 
            self.basis_trial = TensorizedBasis(self.hparams['trial_bases'])
        if self.hparams.get('POD',False)==True:
            self.basis_test = BSplineInterpolatedPOD2D(N_samples=self.hparams['N_samples_train'], d=self.hparams['d'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'], w=self.w_Omega, xi=self.xi_Omega, N=self.hparams['N'], device=self.used_device)
            self.basis_trial = BSplineInterpolatedPOD2D(N_samples=self.hparams['N_samples_train'], d=self.hparams['d'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'], w=self.w_Omega, xi=self.xi_Omega, N=self.hparams['N'], device=self.used_device)
        #Basis evaluation at quadrature points
        self.psix = torch.tensor(self.basis_trial.forward(self.xi_OmegaT_L.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        #A_0 (K inverse for constant theta)
        if self.hparams['Neumannseries']==True:
            self.F_0, self.A_0 = self.compute_F_0_A_0()
        #Identity
        self.Identity = torch.eye(self.hparams['N'], dtype=self.hparams['dtype'], device=self.used_device)
        self = self.to(self.hparams['dtype'])

    def discretize_input_functions(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        theta_d = discretize_functions(theta, self.xi_OmegaT, device=self.used_device)
        theta_x0_d = discretize_functions(theta, self.xi_Gamma_x0, device=self.used_device)
        theta_xL_d = discretize_functions(theta, self.xi_Gamma_xL, device=self.used_device)
        f_d = discretize_functions(f, self.xi_OmegaT, device=self.used_device)
        eta_y0_d = discretize_functions(eta_y0, self.xi_Gamma_y0, device=self.used_device)
        eta_yL_d = discretize_functions(eta_yL, self.xi_Gamma_yL, device=self.used_device)
        g_x0_d = discretize_functions(g_x0, self.xi_Gamma_x0, device=self.used_device)
        g_xL_d = discretize_functions(g_xL, self.xi_Gamma_xL, device=self.used_device)
        u0_d = discretize_functions(u0, self.xi_Gamma_t0, device=self.used_device)
        return theta_d, theta_x0_d, theta_xL_d, f_d, eta_y0_d, eta_yL_d, g_x0_d, g_xL_d, u0_d
    
    def discretize_output_function(self, u):
        u_d = discretize_functions(u, self.xi_OmegaT_L, device=self.used_device)
        return u_d

    def compute_F(self, theta, theta_x0, theta_xL):
        if  self.hparams['modeltype']=='data NGO':
            basis_test = self.basis_test.forward(self.xi_OmegaT.cpu().numpy())
            F = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT.cpu().numpy(), basis_test, theta.cpu().numpy()).reshape(([theta.shape[0]].append(self.hparams['h'])))
        if  self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM':
            basis_trial = self.basis_trial.forward(self.xi_OmegaT.cpu().numpy())
            gradbasis_test = self.basis_test.grad(self.xi_OmegaT.cpu().numpy())[:,:,1:]
            gradbasis_trial = self.basis_trial.grad(self.xi_OmegaT.cpu().numpy())[:,:,1:]
            ddtbasis_test = self.basis_test.grad(self.xi_OmegaT.cpu().numpy())[:,:,0]
            basis_test_x0 = self.basis_test.forward(self.xi_Gamma_x0.cpu().numpy())
            basis_test_xL = self.basis_test.forward(self.xi_Gamma_xL.cpu().numpy())
            gradbasis_test_x0 = self.basis_test.grad(self.xi_Gamma_x0.cpu().numpy())[:,:,1:]
            gradbasis_test_xL = self.basis_test.grad(self.xi_Gamma_xL.cpu().numpy())[:,:,1:]
            basis_trial_x0 = self.basis_trial.forward(self.xi_Gamma_x0.cpu().numpy())
            basis_trial_xL = self.basis_trial.forward(self.xi_Gamma_xL.cpu().numpy())
            gradbasis_trial_x0 = self.basis_trial.grad(self.xi_Gamma_x0.cpu().numpy())[:,:,1:]
            gradbasis_trial_xL = self.basis_trial.grad(self.xi_Gamma_xL.cpu().numpy())[:,:,1:]
            basis_test_tT = self.basis_test.forward(self.xi_Gamma_tT.cpu().numpy())
            basis_trial_tT = self.basis_trial.forward(self.xi_Gamma_tT.cpu().numpy())
            F = opt_einsum.contract('q,Nq,qmx,qnx->Nmn', self.w_OmegaT.cpu().numpy(), theta.cpu().numpy(), gradbasis_test, gradbasis_trial)
            F += - opt_einsum.contract('q,qn,qm->mn', self.w_OmegaT.cpu().numpy(), basis_trial, ddtbasis_test)
            F += -opt_einsum.contract('q,qm,x,Nq,qnx->Nmn', self.w_Gamma_x0.cpu().numpy(), basis_test_x0, self.n_x0.cpu().numpy(), theta_x0.cpu().numpy(), gradbasis_trial_x0)
            F += -opt_einsum.contract('q,qm,x,Nq,qnx->Nmn', self.w_Gamma_xL.cpu().numpy(), basis_test_xL, self.n_xL.cpu().numpy(), theta_xL.cpu().numpy(), gradbasis_trial_xL)
            F += -opt_einsum.contract('q,qn,x,Nq,qmx->Nmn', self.w_Gamma_x0.cpu().numpy(), basis_trial_x0, self.n_x0.cpu().numpy(), theta_x0.cpu().numpy(), gradbasis_test_x0)
            F += -opt_einsum.contract('q,qn,x,Nq,qmx->Nmn', self.w_Gamma_xL.cpu().numpy(), basis_trial_xL, self.n_xL.cpu().numpy(), theta_xL.cpu().numpy(), gradbasis_test_xL)
            F += opt_einsum.contract('q,qm,qn->mn', self.w_Gamma_tT.cpu().numpy(), basis_test_tT, basis_trial_tT)
        if self.hparams.get('gamma_stabilization',0)!=0:
            F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_x0.cpu().numpy(), basis_test_x0, theta_x0.cpu().numpy(), basis_trial_x0)
            F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_xL.cpu().numpy(), basis_test_xL, theta_xL.cpu().numpy(), basis_trial_xL)
        return F
    
    def compute_F_0_A_0(self):
        theta = torch.ones(self.w_OmegaT.shape, dtype=self.hparams['dtype'], device=self.used_device).reshape(1,self.w_OmegaT.shape[0])
        theta_x0 = torch.ones(self.w_Gamma_x0.shape, dtype=self.hparams['dtype'], device=self.used_device).reshape(1,self.w_Gamma_x0.shape[0])
        theta_xL = torch.ones(self.w_Gamma_x0.shape, dtype=self.hparams['dtype'], device=self.used_device).reshape(1,self.w_Gamma_xL.shape[0])
        basis_trial = self.basis_trial.forward(self.xi_OmegaT.cpu().numpy())
        gradbasis_test = self.basis_test.grad(self.xi_OmegaT.cpu().numpy())[:,:,1:]
        gradbasis_trial = self.basis_trial.grad(self.xi_OmegaT.cpu().numpy())[:,:,1:]
        ddtbasis_test = self.basis_test.grad(self.xi_OmegaT.cpu().numpy())[:,:,0]
        basis_test_x0 = self.basis_test.forward(self.xi_Gamma_x0.cpu().numpy())
        basis_test_xL = self.basis_test.forward(self.xi_Gamma_xL.cpu().numpy())
        gradbasis_test_x0 = self.basis_test.grad(self.xi_Gamma_x0.cpu().numpy())[:,:,1:]
        gradbasis_test_xL = self.basis_test.grad(self.xi_Gamma_xL.cpu().numpy())[:,:,1:]
        basis_trial_x0 = self.basis_trial.forward(self.xi_Gamma_x0.cpu().numpy())
        basis_trial_xL = self.basis_trial.forward(self.xi_Gamma_xL.cpu().numpy())
        gradbasis_trial_x0 = self.basis_trial.grad(self.xi_Gamma_x0.cpu().numpy())[:,:,1:]
        gradbasis_trial_xL = self.basis_trial.grad(self.xi_Gamma_xL.cpu().numpy())[:,:,1:]
        basis_test_tT = self.basis_test.forward(self.xi_Gamma_tT.cpu().numpy())
        basis_trial_tT = self.basis_trial.forward(self.xi_Gamma_tT.cpu().numpy())
        F_0 = opt_einsum.contract('q,Nq,qmx,qnx->Nmn', self.w_OmegaT.cpu().numpy(), theta.cpu().numpy(), gradbasis_test, gradbasis_trial)
        F_0 += - opt_einsum.contract('q,qn,qm->mn', self.w_OmegaT.cpu().numpy(), basis_trial, ddtbasis_test)
        F_0 += -opt_einsum.contract('q,qm,x,Nq,qnx->Nmn', self.w_Gamma_x0.cpu().numpy(), basis_test_x0, self.n_x0.cpu().numpy(), theta_x0.cpu().numpy(), gradbasis_trial_x0)
        F_0 += -opt_einsum.contract('q,qm,x,Nq,qnx->Nmn', self.w_Gamma_xL.cpu().numpy(), basis_test_xL, self.n_xL.cpu().numpy(), theta_xL.cpu().numpy(), gradbasis_trial_xL)
        F_0 += -opt_einsum.contract('q,qn,x,Nq,qmx->Nmn', self.w_Gamma_x0.cpu().numpy(), basis_trial_x0, self.n_x0.cpu().numpy(), theta_x0.cpu().numpy(), gradbasis_test_x0)
        F_0 += -opt_einsum.contract('q,qn,x,Nq,qmx->Nmn', self.w_Gamma_xL.cpu().numpy(), basis_trial_xL, self.n_xL.cpu().numpy(), theta_xL.cpu().numpy(), gradbasis_test_xL)
        F_0 += opt_einsum.contract('q,qm,qn->mn', self.w_Gamma_tT.cpu().numpy(), basis_test_tT, basis_trial_tT)
        if self.hparams.get('gamma_stabilization',0)!=0:
            F_0 += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_x0.cpu().numpy(), basis_test_x0, theta_x0.cpu().numpy(), basis_trial_x0)
            F_0 += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_xL.cpu().numpy(), basis_test_xL, theta_xL.cpu().numpy(), basis_trial_xL)
        F_0 = F_0[0]
        F_0 = torch.tensor(F_0, dtype=self.hparams['dtype'], device=self.used_device)
        A_0 = torch.linalg.pinv(F_0)
        A_0 = torch.tensor(A_0, dtype=self.hparams['dtype'], device=self.used_device)
        if  self.hparams['modeltype']=='data NGO':
            basis_test = self.basis_test.forward(self.xi_OmegaT.cpu().numpy())
            F = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT.cpu().numpy(), basis_test, theta.cpu().numpy()).reshape((theta.shape[0],self.hparams['h'][0],self.hparams['h'][1]))
            F_0 = torch.tensor(F, dtype=self.hparams['dtype'], device=self.used_device)
        return F_0, A_0
    
    def compute_d(self, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        basis_test = self.basis_test.forward(self.xi_OmegaT.cpu().numpy())
        basis_test_y0 = self.basis_test.forward(self.xi_Gamma_y0.cpu().numpy())
        basis_test_yL = self.basis_test.forward(self.xi_Gamma_yL.cpu().numpy())
        basis_test_x0 = self.basis_test.forward(self.xi_Gamma_x0.cpu().numpy())
        basis_test_xL = self.basis_test.forward(self.xi_Gamma_xL.cpu().numpy())
        gradbasis_test_x0 = self.basis_test.grad(self.xi_Gamma_x0.cpu().numpy())[:,:,1:]
        gradbasis_test_xL = self.basis_test.grad(self.xi_Gamma_xL.cpu().numpy())[:,:,1:]
        basis_test_t0 = self.basis_test.forward(self.xi_Gamma_t0.cpu().numpy())
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT.cpu().numpy(), basis_test, f.cpu().numpy())
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_y0.cpu().numpy(), basis_test_y0, eta_y0.cpu().numpy())
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_yL.cpu().numpy(), basis_test_yL, eta_yL.cpu().numpy())
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_x0.cpu().numpy(), self.n_x0.cpu().numpy(), gradbasis_test_x0, g_x0.cpu().numpy())
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_xL.cpu().numpy(), self.n_xL.cpu().numpy(), gradbasis_test_xL, g_xL.cpu().numpy())
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_t0.cpu().numpy(), basis_test_t0, u0.cpu().numpy())
        if self.hparams.get('gamma_stabilization',0)!=0:
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_x0.cpu().numpy(), basis_test_x0, g_x0.cpu().numpy())
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_xL.cpu().numpy(), basis_test_xL, g_xL.cpu().numpy())        
        return d
    
    def NN_forward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        theta = theta.reshape((theta.shape[0],self.hparams['Q'],self.hparams['Q']))
        f = f.reshape((f.shape[0],self.hparams['Q'],self.hparams['Q']))
        eta = torch.zeros((eta_y0.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        eta[:,:,0] = eta_y0
        eta[:,:,-1] = eta_yL
        g = torch.zeros((g_x0.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        g[:,0,:] = g_x0
        g[:,-1,:] = g_xL
        inputfuncs = torch.stack((theta,f,eta,g), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta.flatten(-2,-1),f.flatten(-2,-1),eta_y0,eta_yL,g_x0,g_xL), dim=1)
        u_hat = self.systemnet.forward(inputfuncs).reshape((theta.shape[0],self.hparams['Q_L']**self.hparams['d']))
        return u_hat
    
    def DeepONet_forward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        theta = theta.reshape((theta.shape[0],self.hparams['Q'],self.hparams['Q']))
        f = f.reshape((f.shape[0],self.hparams['Q'],self.hparams['Q']))
        eta = torch.zeros((eta_y0.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        eta[:,:,0] = eta_y0
        eta[:,:,-1] = eta_yL
        g = torch.zeros((g_x0.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        g[:,0,:] = g_x0
        g[:,-1,:] = g_xL
        inputfuncs = torch.stack((theta,f,eta,g), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta.flatten(-2,-1),f.flatten(-2,-1),eta_y0,eta_yL,g_x0,g_xL), dim=1)
        u_n = self.systemnet.forward(inputfuncs).reshape((theta.shape[0],self.hparams['N']))
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat
    
    def VarMiON_forward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL,):
        eta = torch.zeros((eta_y0.shape[0],2*eta_y0.shape[1]), dtype=self.hparams['dtype'], device=self.used_device)
        eta[:,:eta_y0.shape[1]] = eta_y0
        eta[:,eta_y0.shape[1]:] = eta_yL
        g = torch.zeros((g_x0.shape[0],2*g_x0.shape[1]), dtype=self.hparams['dtype'], device=self.used_device)
        g[:,:g_x0.shape[1]] = g_x0
        g[:,g_x0.shape[1]:] = g_xL
        systemnet = self.systemnet.forward(theta)
        LBranch = self.LBranch_f.forward(f) + self.LBranch_eta.forward(eta) + self.LBranch_g.forward(g)
        u_n = opt_einsum.contract('nij,nj->ni', systemnet, LBranch)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat

    def NGO_forward(self, theta_bar, F, d):
        if self.hparams.get('scaling_equivariance',False)==True:
            F = F/theta_bar[:,None,None]
        if self.hparams.get('Neumannseries', False)==False:
            A = self.systemnet.forward(F)
        if self.hparams.get('Neumannseries', False)==True:
            A_0 = self.A_0 if self.hparams.get('A0net')==None else self.A0net.systemnet.forward(F)
            T = torch.zeros(F.shape, dtype=self.hparams['dtype'], device=self.used_device)
            Ti = self.Identity
            T1 = -F@A_0 + self.Identity
            for i in range(0, self.hparams['Neumannseries_order']):
                Ti = T1@Ti
                T = T + Ti
            A = A_0@(self.Identity + T + self.systemnet.forward(T1))
        if self.hparams.get('scaling_equivariance',False)==True:
            A = A/theta_bar[:,None,None]        
        u_n = opt_einsum.contract('nij,nj->ni', A, d)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat

    def FEM_forward(self, F, d):
        if self.hparams.get('Neumannseries', False)==False:
            K_inv = torch.linalg.pinv(F)
        if self.hparams.get('Neumannseries', False)==True:
            T = torch.zeros(F.shape, dtype=self.hparams['dtype'], device=self.used_device)
            Ti = self.Identity
            T1 = -F@self.A_0 + self.Identity
            for i in range(0, self.hparams['Neumannseries_order']):
                Ti = T1@Ti
                T = T + Ti
            K_inv = self.A_0@(self.Identity + T)
        u_n = opt_einsum.contract('nij,nj->ni', K_inv, d)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat
    
    def projection_forward(self, u):
        u_q = torch.tensor(discretize_functions(u, self.xi_Omega, device=self.used_device), dtype=self.hparams['dtype'])
        basis_test = torch.tensor(self.basis_test.forward(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        basis_trial = torch.tensor(self.basis_trial.forward(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        u_w = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, u_q)
        M = opt_einsum.contract('q,qm,qn->mn', self.w_Omega, basis_test, basis_trial)
        M_inv = torch.linalg.pinv(M)
        u_n = opt_einsum.contract('mn,Nm->Nn', M_inv, u_w)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat
    
    def init_modeltype(self):
        if self.hparams['modeltype']=='NN':
            self.forwardfunction = self.NN_forward
        if self.hparams['modeltype']=='DeepONet':
            self.forwardfunction = self.DeepONet_forward
        if self.hparams['modeltype']=='VarMiON':
            self.forwardfunction = self.VarMiON_forward
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO':
            self.forwardfunction = self.NGO_forward
        if self.hparams['modeltype']=='FEM':
            self.forwardfunction = self.FEM_forward
        if self.hparams['modeltype']=='projection':
            self.forwardfunction = self.projection_forward
    
    def forward(self, *args):
        u_hat = self.forwardfunction(*args)
        return u_hat
    
    def simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, x, u):
        self.geometry()
        #self.compute_F_0_A_0()
        theta, theta_g, f, eta_y0, eta_yL, g_x0, g_xL = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO' or self.hparams['modeltype']=='FEM':
            F = torch.tensor(self.compute_F(theta, theta_g), dtype=self.hparams['dtype'], device=self.used_device)
            d = torch.tensor(self.compute_d(f, eta_y0, eta_yL, g_x0, g_xL), dtype=self.hparams['dtype'], device=self.used_device)
        theta = torch.tensor(theta, dtype=self.hparams['dtype'], device=self.used_device)
        f = torch.tensor(f, dtype=self.hparams['dtype'], device=self.used_device)
        eta_y0 = torch.tensor(eta_y0, dtype=self.hparams['dtype'], device=self.used_device)
        eta_yL = torch.tensor(eta_yL, dtype=self.hparams['dtype'], device=self.used_device)
        g_x0 = torch.tensor(g_x0, dtype=self.hparams['dtype'], device=self.used_device)
        gr = torch.tensor(gr, dtype=self.hparams['dtype'], device=self.used_device)    
        self.psix = torch.tensor(self.basis_trial.forward(x.cpu()), dtype=self.hparams['dtype'], device=self.used_device)
        if self.hparams['modeltype']=='NN':
            u_hat = self.NN_forward(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        if self.hparams['modeltype']=='DeepONet':
            u_hat = self.DeepONet_forward(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        if self.hparams['modeltype']=='VarMiON':
            u_hat = self.VarMiON_forward(theta, f, eta_y0, eta_yL, g_x0, g_xL)
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO':
            theta_bar = torch.tensor(torch.sum(self.w_Omega[None,:]*theta, axis=-1), dtype=self.hparams['dtype'], device=self.used_device)
            u_hat = self.NGO_forward(theta_bar, F, d)
        if self.hparams['modeltype']=='FEM':
            u_hat = self.FEM_forward(F, d)
        if self.hparams['modeltype']=='projection':
            u_hat = self.projection_forward(u)
        return u_hat
    
    def geometry(self):
        #Quadrature
        if self.hparams['quadrature']=='Gauss-Legendre':
            quad_OmegaT = GaussLegendreQuadrature(Q=self.hparams['Q'], n_elements=self.hparams['n_elements'])
            quad_Gamma_t = GaussLegendreQuadrature(Q=[self.hparams['Q'][1],self.hparams['Q'][2]], n_elements=[self.hparams['n_elements'][1],self.hparams['Q'][2]])
            quad_Gamma_x = GaussLegendreQuadrature(Q=[self.hparams['Q'][0],self.hparams['Q'][2]], n_elements=[self.hparams['n_elements'][0],self.hparams['Q'][2]])
            quad_Gamma_y = GaussLegendreQuadrature(Q=[self.hparams['Q'][0],self.hparams['Q'][1]], n_elements=[self.hparams['n_elements'][0],self.hparams['Q'][1]])

            self.w_OmegaT = quad_OmegaT.w
            self.xi_OmegaT = quad_OmegaT.xi
            
            self.w_Gamma_t0 = quad_Gamma_t.w
            self.xi_Gamma_t0 = np.zeros((self.hparams['Q'][1]*self.hparams['Q'][2],self.hparams['d']))

            self.xi_Gamma_t0[:,1] = quad_Gamma_t.xi[:,0]
            self.xi_Gamma_t0[:,2] = quad_Gamma_t.xi[:,1]

            self.w_Gamma_tT = quad_Gamma_t.w
            self.xi_Gamma_tT = np.ones((self.hparams['Q'][1]*self.hparams['Q'][2],self.hparams['d']))
            self.xi_Gamma_tT[:,1] = quad_Gamma_t.xi[:,0]
            self.xi_Gamma_tT[:,2] = quad_Gamma_t.xi[:,1]

            self.w_Gamma_x0 = quad_Gamma_x.w
            self.xi_Gamma_x0 = np.zeros((self.hparams['Q'][0]*self.hparams['Q'][2],self.hparams['d']))
            self.xi_Gamma_x0[:,0] = quad_Gamma_x.xi[:,0]
            self.xi_Gamma_x0[:,2] = quad_Gamma_x.xi[:,1]

            self.w_Gamma_xL = quad_Gamma_x.w
            self.xi_Gamma_xL = np.ones((self.hparams['Q'][0]*self.hparams['Q'][2],self.hparams['d']))
            self.xi_Gamma_x0[:,0] = quad_Gamma_x.xi[:,0]
            self.xi_Gamma_x0[:,2] = quad_Gamma_x.xi[:,1]

            self.w_Gamma_y0 = quad_Gamma_y.w
            self.xi_Gamma_y0 = np.zeros((self.hparams['Q'][0]*self.hparams['Q'][1],self.hparams['d']))
            self.xi_Gamma_y0[:,0] = quad_Gamma_y.xi[:,0]
            self.xi_Gamma_y0[:,1] = quad_Gamma_y.xi[:,1]

            self.w_Gamma_yL = quad_Gamma_y.w
            self.xi_Gamma_yL = np.ones((self.hparams['Q'][0]*self.hparams['Q'][1],self.hparams['d']))
            self.xi_Gamma_y0[:,0] = quad_Gamma_y.xi[:,0]
            self.xi_Gamma_y0[:,1] = quad_Gamma_y.xi[:,1]

            self.w_OmegaT = torch.tensor(self.w_OmegaT, dtype=self.hparams['dtype'], device=self.used_device)
            self.w_Gamma_t0 = torch.tensor(self.w_Gamma_t0, dtype=self.hparams['dtype'], device=self.used_device)
            self.w_Gamma_tT = torch.tensor(self.w_Gamma_tT, dtype=self.hparams['dtype'], device=self.used_device)
            self.w_Gamma_y0 = torch.tensor(self.w_Gamma_y0, dtype=self.hparams['dtype'], device=self.used_device)
            self.w_Gamma_yL = torch.tensor(self.w_Gamma_yL, dtype=self.hparams['dtype'], device=self.used_device)
            self.w_Gamma_x0 = torch.tensor(self.w_Gamma_x0, dtype=self.hparams['dtype'], device=self.used_device)
            self.w_Gamma_xL = torch.tensor( self.w_Gamma_xL, dtype=self.hparams['dtype'], device=self.used_device)

            self.xi_OmegaT = torch.tensor(self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.used_device)
            self.xi_Gamma_t0 = torch.tensor(self.xi_Gamma_t0, dtype=self.hparams['dtype'], device=self.used_device)
            self.xi_Gamma_tT = torch.tensor(self.xi_Gamma_tT, dtype=self.hparams['dtype'], device=self.used_device)
            self.xi_Gamma_y0 = torch.tensor(self.xi_Gamma_y0, dtype=self.hparams['dtype'], device=self.used_device)
            self.xi_Gamma_yL = torch.tensor(self.xi_Gamma_yL, dtype=self.hparams['dtype'], device=self.used_device)
            self.xi_Gamma_x0 = torch.tensor(self.xi_Gamma_x0, dtype=self.hparams['dtype'], device=self.used_device)
            self.xi_Gamma_xL = torch.tensor(self.xi_Gamma_xL, dtype=self.hparams['dtype'], device=self.used_device)

        #Outward normal
        self.n_y0 = torch.tensor([0,-1], dtype=self.hparams['dtype'], device=self.used_device)
        self.n_yL = torch.tensor([0,1], dtype=self.hparams['dtype'], device=self.used_device)
        self.n_x0 = torch.tensor([-1,0], dtype=self.hparams['dtype'], device=self.used_device)
        self.n_xL = torch.tensor([1,0], dtype=self.hparams['dtype'], device=self.used_device)

        #Loss quadrature
        if self.hparams['quadrature_L']=='Gauss-Legendre':
            quad_OmegaT_L = GaussLegendreQuadrature(Q=self.hparams['Q_L'], n_elements=self.hparams['n_elements_L'])
        self.w_OmegaT_L = torch.tensor(quad_OmegaT_L.w, dtype=self.hparams['dtype'], device=self.used_device)
        self.xi_OmegaT_L = torch.tensor(quad_OmegaT_L.xi, dtype=self.hparams['dtype'], device=self.used_device)

    def configure_optimizers(self):
        self.metric = [1]
        self.automatic_optimization = False
        self.optimizer_idx = 0
        optimizer1 = self.hparams['optimizer1'](self.parameters(), lr=self.hparams['learning_rate'])
        optimizer2 = self.hparams['optimizer2'](self.parameters(), lr=1, line_search_fn='strong_wolfe', history_size=200, max_iter=100)
        return [optimizer1, optimizer2]

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()[self.optimizer_idx]
        inputs = train_batch[:-1]
        def closure():
            opt.zero_grad()
            loss = 0
            if self.hparams.get('solution_loss',None)!=None:
                u = train_batch[-1]
                u_hat = self.forward(*inputs)
                loss += self.hparams['solution_loss'](self.w_OmegaT_L, u_hat, u)
            if self.hparams.get('matrix_loss',None)!=None:
                F = inputs[1]
                d = inputs[2]
                if self.hparams.get('Neumannseries', False)==False:
                    A_hat = self.systemnet.forward(F)
                if self.hparams.get('Neumannseries', False)==True:
                    A_0 = self.A_0 if self.hparams.get('A0net')==None else self.A0net.systemnet.forward(F)
                    T = torch.zeros(F.shape, dtype=self.hparams['dtype'], device=self.used_device)
                    Ti = self.Identity
                    T1 = - (F - self.F_0)@A_0
                    for i in range(0, self.hparams['Neumannseries_order']):
                        Ti = T1@Ti
                        T = T + Ti
                    A_hat = A_0@(self.Identity + T + self.systemnet.forward(T1))
                loss += self.hparams['matrix_loss'](torch.matmul(F, torch.matmul(A_hat,F)), F)
                # loss += self.hparams['matrix_loss'](torch.matmul(A_hat, torch.matmul(F,A_hat)), A_hat)
                # u_hat_1 = opt_einsum.contract('nij,njk,nkl,nl->ni', A_hat,F,A_hat,d)
                # u_hat_2 = opt_einsum.contract('nij,nj->ni', A_hat,d)
                # loss += self.hparams['matrix_loss'](u_hat_1,u_hat_2)
            self.manual_backward(loss)
            self.log('train_loss', loss)
            return loss
        opt.step(closure=closure)

    def validation_step(self, val_batch, batch_idx):
        inputs = val_batch[:-1]
        u = val_batch[-1]
        u_hat = self.forward(*inputs)
        loss = 0
        if self.hparams.get('solution_loss',None)!=None:
            loss += self.hparams['solution_loss'](self.w_OmegaT_L, u_hat, u)
        if self.hparams.get('matrix_loss',None)!=None:
            F = inputs[1]
            d = inputs[2]
            if self.hparams.get('Neumannseries',False)==False:
                A_hat = self.systemnet.forward(F)
            if self.hparams.get('Neumannseries',False)==True:
                A_0 = self.A_0 if self.hparams.get('A0net')==None else self.A0net.systemnet.forward(F)
                T = torch.zeros(F.shape, dtype=self.hparams['dtype'], device=self.used_device)
                Ti = self.Identity
                T1 = - (F - self.F_0)@A_0
                for i in range(0, self.hparams['Neumannseries_order']):
                    Ti = T1@Ti
                    T = T + Ti
                A_hat = A_0@(self.Identity + T + self.systemnet.forward(T1))
            loss += self.hparams['matrix_loss'](torch.matmul(F, torch.matmul(A_hat,F)), F)
            # loss += self.hparams['matrix_loss'](torch.matmul(A_hat, torch.matmul(F,A_hat)), A_hat)
            # u_hat_1 = opt_einsum.contract('nij,njk,nkl,nl->ni', A_hat,F,A_hat,d)
            # u_hat_2 = opt_einsum.contract('nij,nj->ni', A_hat,d)
            # loss += self.hparams['matrix_loss'](u_hat_1,u_hat_2)
        metric = self.hparams['metric'](self.w_OmegaT_L, u_hat, u)
        self.metric.append(metric)
        self.log('val_loss', loss)
        self.log('metric', metric)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['hparams'] = self.hparams
        
    def on_fit_start(self):
        self.used_device = self.device
        torch.set_num_threads(2)
        self.psix = self.psix.to(self.used_device)
        self.w_OmegaT_L = torch.tensor(self.w_OmegaT_L).to(self.used_device)
        self.systemnet.device = self.used_device
        if self.hparams.get('A0net')!=None:
            self.A0net = loadmodelfromlabel(model=NeuralOperator, label=self.hparams['A0net'][1], logdir='../../../nnlogs', sublogdir=self.hparams['A0net'][0], map_location=self.hparams['used_device'])
            self.A0net.systemnet = self.A0net.systemnet.to(self.used_device)
            for param in self.A0net.parameters():
                param.requires_grad = False

    def on_validation_epoch_end(self):
        print(self.metric[-1])
        if self.hparams['switch_threshold']!=None:
            if self.metric[-1]<self.hparams['switch_threshold']:
                    self.optimizer_idx = 1
                    self.hparams['batch_size'] = self.hparams['N_samples']