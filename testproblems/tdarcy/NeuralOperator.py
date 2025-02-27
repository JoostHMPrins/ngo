import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
import numpy as np
import opt_einsum

import sys
sys.path.insert(0, '/home/prins/st8/prins/phd/git/ngo-pde-gk/ml')
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
        self = self.to(self.hparams['dtype'])
        #compute_quadrature and quadrature
        self.compute_quadrature()
        #Model type
        self.init_modeltype()
        #Linear branches
        if self.hparams['modeltype']=='VarMiON':
            self.LBranch_f = LBranchNet(hparams, input_dim=len(self.xi_OmegaT), output_dim=self.hparams['N']).to(self.used_device)
            self.LBranch_eta = LBranchNet(hparams, input_dim=len(self.xi_Gamma_y0)+len(self.xi_Gamma_yL), output_dim=self.hparams['N']).to(self.used_device)
            self.LBranch_g = LBranchNet(hparams, input_dim=len(self.xi_Gamma_x0)+len(self.xi_Gamma_xL), output_dim=self.hparams['N']).to(self.used_device)
            self.LBranch_u0 = LBranchNet(hparams, input_dim=len(self.xi_Gamma_t0), output_dim=self.hparams['N']).to(self.used_device)
        self.hparams['N_w_real'] = sum(p.numel() for p in self.parameters())
        #System net
        self.systemnet = self.hparams['systemnet'](self.hparams)
        self.systemnet = self.systemnet.to(self.used_device)
        #Bases
        if self.hparams.get('POD',False)==False:
            self.basis_test = TensorizedBasis(self.hparams['test_bases']) 
            self.basis_trial = TensorizedBasis(self.hparams['trial_bases'])
        if self.hparams.get('POD',False)==True:
            self.basis_test = BSplineInterpolatedPOD2D(N_samples=self.hparams['N_samples_train'], d=self.hparams['d'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'], w=self.w_Omega, xi=self.xi_Omega, N=self.hparams['N'], device=self.used_device)
            self.basis_trial = BSplineInterpolatedPOD2D(N_samples=self.hparams['N_samples_train'], d=self.hparams['d'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'], w=self.w_Omega, xi=self.xi_Omega, N=self.hparams['N'], device=self.used_device)
        #Relevant matrices
        if self.hparams['modeltype']!='NN':
            self.Identity = torch.eye(self.hparams['N'], dtype=self.hparams['dtype'], device=self.used_device)
            self.compute_basis_evaluations()
            # self.rescale_quadrature_t()
            self.compute_Kronecker_factors_x()
            self.compute_Kronecker_factors_t()
            if self.hparams['Neumannseries']==True:
                self.A_0 = self.compute_A_0()
        
    def compute_basis_evaluations(self):
        zero = np.zeros((1,1))
        one = np.ones((1,1))
        self.basis_test_OmegaT = self.basis_test.forward(self.xi_OmegaT)
        self.basis_test_Gamma_y0 = self.basis_test.forward(self.xi_Gamma_y0)
        self.basis_test_Gamma_yL = self.basis_test.forward(self.xi_Gamma_yL)
        self.basis_test_Gamma_x0 = self.basis_test.forward(self.xi_Gamma_x0)
        self.basis_test_Gamma_xL = self.basis_test.forward(self.xi_Gamma_xL)
        self.gradbasis_test_Gamma_x0 = self.basis_test.grad(self.xi_Gamma_x0)[:,:,1:]
        self.gradbasis_test_Gamma_xL = self.basis_test.grad(self.xi_Gamma_xL)[:,:,1:]
        self.basis_test_Gamma_t0 = self.basis_test.forward(self.xi_Gamma_t0)
        self.basis_trial_Gamma_t0 = self.basis_trial.forward(self.xi_Gamma_t0)
        self.basis_trial_Gamma_tT = self.basis_trial.forward(self.xi_Gamma_tT)
        self.gradbasis_trial_Gamma_x0 = self.basis_trial.grad(self.xi_Gamma_x0)[:,:,1:]
        self.gradbasis_trial_Gamma_xL = self.basis_trial.grad(self.xi_Gamma_xL)[:,:,1:]
        self.basis_test_t = self.hparams['test_bases'][0].forward(self.xi_t)
        self.gradbasis_test_t = self.hparams['test_bases'][0].grad(self.xi_t)
        self.basis_trial_t = self.hparams['trial_bases'][0].forward(self.xi_t)
        # self.gradbasis_trial_t = self.hparams['trial_bases'][0].grad(self.xi_t), dtype=self.hparams['dtype'], device=self.used_device)
        self.basis_test_t0 = self.hparams['test_bases'][0].forward(zero)[0]
        # self.gradbasis_test_t0 = self.hparams['test_bases'][0].grad(t0)[0], dtype=self.hparams['dtype'], device=self.used_device)
        # self.basis_trial_t0 = self.hparams['trial_bases'][0].forward(t0)[0], dtype=self.hparams['dtype'], device=self.used_device)
        # self.gradbasis_trial_t0 = self.hparams['trial_bases'][0].grad(t0)[0], dtype=self.hparams['dtype'], device=self.used_device)
        self.basis_test_tT = self.hparams['test_bases'][0].forward(one)[0]
        # self.gradbasis_test_tT = self.hparams['test_bases'][0].grad(t1)[0], dtype=self.hparams['dtype'], device=self.used_device)
        self.basis_trial_tT = self.hparams['trial_bases'][0].forward(one)[0]

    def compute_Kronecker_factors_x(self):
        zero = np.zeros((1,1))
        one = np.ones((1,1))
        basis_test_x = self.hparams['test_bases'][1].forward(self.xi_x)
        basis_test_y = self.hparams['test_bases'][2].forward(self.xi_y)
        gradbasis_test_x = self.hparams['test_bases'][1].grad(self.xi_x)
        gradbasis_test_y = self.hparams['test_bases'][2].grad(self.xi_y)   
        basis_trial_x = self.hparams['trial_bases'][1].forward(self.xi_x)
        basis_trial_y = self.hparams['trial_bases'][2].forward(self.xi_y)
        gradbasis_trial_x = self.hparams['trial_bases'][1].grad(self.xi_x)
        gradbasis_trial_y = self.hparams['trial_bases'][2].grad(self.xi_y)
        basis_test_x0 = self.hparams['test_bases'][1].forward(zero)[0]
        basis_test_y0 = self.hparams['test_bases'][2].forward(zero)[0]
        gradbasis_test_x0 = self.hparams['test_bases'][1].grad(zero)[0]
        # gradbasis_test_y0 = self.hparams['test_bases'][2].grad(zero)[0], dtype=self.hparams['dtype'], device=self.used_device)
        basis_trial_x0 = self.hparams['trial_bases'][1].forward(zero)[0]
        # basis_trial_y0 = self.hparams['trial_bases'][2].forward(zero)[0], dtype=self.hparams['dtype'], device=self.used_device)
        gradbasis_trial_x0 = self.hparams['trial_bases'][1].grad(zero)[0]
        # gradbasis_trial_y0 = self.hparams['trial_bases'][2].grad(zero)[0], dtype=self.hparams['dtype'], device=self.used_device)
        basis_test_xL = self.hparams['test_bases'][1].forward(one)[0]
        basis_test_yL = self.hparams['test_bases'][2].forward(one)[0]
        gradbasis_test_xL = self.hparams['test_bases'][1].grad(one)[0]
        # gradbasis_test_yL = self.hparams['test_bases'][2].grad(one)[0], dtype=self.hparams['dtype'], device=self.used_device)
        basis_trial_xL = self.hparams['trial_bases'][1].forward(one)[0]
        # basis_trial_yL = self.hparams['trial_bases'][2].forward(one)[0], dtype=self.hparams['dtype'], device=self.used_device)
        gradbasis_trial_xL = self.hparams['trial_bases'][1].grad(one)[0]
        # gradbasis_trial_yL = self.hparams['trial_bases'][2].grad(one)[0], dtype=self.hparams['dtype'], device=self.used_device)
        self.K_x_psipsiphi = opt_einsum.contract('q,qb,qe,qh->beh', self.w_x, basis_test_x, basis_test_x, basis_trial_x)
        self.K_y_psipsiphi = opt_einsum.contract('q,qc,qf,qi->cfi', self.w_y, basis_test_y, basis_test_y, basis_trial_y)
        self.K_x_psidpsidphi = opt_einsum.contract('q,qb,qe,qh->beh', self.w_x, basis_test_x, gradbasis_test_x, gradbasis_trial_x)
        self.K_y_psidpsidphi = opt_einsum.contract('q,qc,qf,qi->cfi', self.w_y, basis_test_y, gradbasis_test_y, gradbasis_trial_y)
        self.K_x_psipsidphi_x0 = opt_einsum.contract('b,e,h->beh', basis_test_x0, basis_test_x0, gradbasis_trial_x0)
        self.K_x_psidpsiphi_x0 = opt_einsum.contract('b,e,h->beh', basis_test_x0, gradbasis_test_x0, basis_trial_x0)
        self.K_x_psipsidphi_xL = opt_einsum.contract('b,e,h->beh', basis_test_xL, basis_test_xL, gradbasis_trial_xL)
        self.K_x_psidpsiphi_xL = opt_einsum.contract('b,e,h->beh', basis_test_xL, gradbasis_test_xL, basis_trial_xL)
        self.K_x_psiphi = opt_einsum.contract('q,qe,qh->eh', self.w_x, basis_test_x, basis_trial_x)
        self.K_y_psiphi = opt_einsum.contract('q,qf,qi->fi', self.w_y, basis_test_y, basis_trial_y)
        self.M_x_psipsi = opt_einsum.contract('q,qb,qe->be', self.w_x, basis_test_x, basis_test_x)
        self.M_y_psipsi = opt_einsum.contract('q,qc,qf->cf', self.w_y, basis_test_y, basis_test_y)
        self.M_y_psipsi_y0 = opt_einsum.contract('c,f->cf', basis_test_y0, basis_test_y0)
        self.M_y_psipsi_yL = opt_einsum.contract('c,f->cf', basis_test_yL, basis_test_yL)
        self.M_x_psidpsi = opt_einsum.contract('q,qb,qe->be', self.w_x, basis_test_x, gradbasis_test_x)
        self.M_x_psidpsi_x0 = opt_einsum.contract('b,e->be', basis_test_x0, gradbasis_test_x0)
        self.M_x_psidpsi_xL = opt_einsum.contract('b,e->be', basis_test_xL, gradbasis_test_xL)
        self.M_x_psipsi = opt_einsum.contract('q,qb,qh->bh', self.w_x, basis_test_x, basis_test_x)
        self.M_y_psipsi = opt_einsum.contract('q,qc,qi->ci', self.w_y, basis_test_y, basis_test_y)
        self.M_x_psipsi_inv = np.linalg.pinv(self.M_x_psipsi)
        self.M_y_psipsi_inv = np.linalg.pinv(self.M_y_psipsi)
        self.M_x_phiphi = opt_einsum.contract('q,qb,qh->bh', self.w_x, basis_trial_x, basis_trial_x)
        self.M_y_phiphi = opt_einsum.contract('q,qc,qi->ci', self.w_y, basis_trial_y, basis_trial_y)
        self.M_x_phiphi_inv = np.linalg.pinv(self.M_x_phiphi)
        self.M_y_phiphi_inv = np.linalg.pinv(self.M_y_phiphi)

    def compute_Kronecker_factors_t(self):
        # zero = np.zeros((1,1))
        # one = np.ones((1,1))
        # basis_test_t = self.hparams['test_bases'][0].forward(self.xi_t)
        # gradbasis_test_t = self.hparams['test_bases'][0].grad(self.xi_t)
        # basis_trial_t = self.hparams['trial_bases'][0].forward(self.xi_t)
        # gradbasis_trial_t = self.hparams['trial_bases'][0].grad(self.xi_t), dtype=self.hparams['dtype'], device=self.used_device)
        # basis_test_t0 = self.hparams['test_bases'][0].forward(zero)[0]
        # gradbasis_test_t0 = self.hparams['test_bases'][0].grad(t0)[0], dtype=self.hparams['dtype'], device=self.used_device)
        # basis_trial_t0 = self.hparams['trial_bases'][0].forward(t0)[0], dtype=self.hparams['dtype'], device=self.used_device)
        # gradbasis_trial_t0 = self.hparams['trial_bases'][0].grad(t0)[0], dtype=self.hparams['dtype'], device=self.used_device)
        # basis_test_tT = self.hparams['test_bases'][0].forward(one)[0]
        # gradbasis_test_tT = self.hparams['test_bases'][0].grad(t1)[0], dtype=self.hparams['dtype'], device=self.used_device)
        # basis_trial_tT = self.hparams['trial_bases'][0].forward(one)[0]
        # gradbasis_trial_tT = self.hparams['trial_bases'][0].grad(t1)[0], dtype=self.hparams['dtype'], device=self.used_device)
        self.K_t_psipsiphi = opt_einsum.contract('q,qa,qd,qg->adg', self.w_t, self.basis_test_t, self.basis_test_t, self.basis_trial_t)
        self.K_t_dpsiphi = opt_einsum.contract('q,qd,qg->dg', self.w_t, self.gradbasis_test_t, self.basis_trial_t)
        self.K_t_psiphi_tT = opt_einsum.contract('d,g->dg', self.basis_test_tT, self.basis_trial_tT)
        self.M_t_psipsi = opt_einsum.contract('q,qa,qd->ad', self.w_t, self.basis_test_t, self.basis_test_t)
        self.M_t_psipsi_t0 = opt_einsum.contract('a,d->ad', self.basis_test_t0, self.basis_test_t0)
        self.M_t_psipsi = opt_einsum.contract('q,qa,qg->ag', self.w_t, self.basis_test_t, self.basis_test_t)
        self.M_t_psipsi_inv = np.linalg.pinv(self.M_t_psipsi)
        self.M_t_phiphi = opt_einsum.contract('q,qa,qg->ag', self.w_t, self.basis_trial_t, self.basis_trial_t)
        self.M_t_phiphi_inv = np.linalg.pinv(self.M_t_phiphi)
        self.M_phiphi = np.kron(np.kron(self.M_t_phiphi, self.M_x_phiphi), self.M_y_phiphi)

    def discretize_input_functions(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        theta_q = discretize_functions(theta, self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        theta_x0_q = discretize_functions(theta, self.xi_Gamma_x0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        theta_xL_q = discretize_functions(theta, self.xi_Gamma_xL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        f_q = discretize_functions(f, self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        eta_y0_q = discretize_functions(eta_y0, self.xi_Gamma_y0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        eta_yL_q = discretize_functions(eta_yL, self.xi_Gamma_yL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        g_x0_q = discretize_functions(g_x0, self.xi_Gamma_x0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        g_xL_q = discretize_functions(g_xL, self.xi_Gamma_xL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        u0_t0_q = discretize_functions(u0, self.xi_Gamma_t0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        return theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q
    
    def discretize_output_function(self, u):
        u_q = discretize_functions(u, self.xi_OmegaT_L, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        return u_q
    
    def project_input_function(self, u_q):
        # u_q = discretize_functions(u, self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT, self.basis_test_OmegaT, u_q)
        M_inv = np.kron(np.kron(self.M_t_psipsi_inv, self.M_x_psipsi_inv), self.M_y_psipsi_inv)
        u_m = opt_einsum.contract('mn,Nm->Nn', M_inv, d)
        return u_m
    
    def project_output_function(self, u_q):
        basis_trial_OmegaT_L = self.basis_trial.forward(self.xi_OmegaT_L)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT_L, basis_trial_OmegaT_L, u_q)
        M_inv = np.kron(np.kron(self.M_t_phiphi_inv, self.M_x_phiphi_inv), self.M_y_phiphi_inv)
        u_m = opt_einsum.contract('mn,Nm->Nn', M_inv, d)
        return u_m

    def compute_F_direct(self, theta_q, theta_x0_q, theta_xL_q):
        # theta = discretize_functions(theta, self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        if  self.hparams['modeltype']=='data NGO':
            # basis_test = torch.tensor(self.basis_test.forward(self.xi_OmegaT), dtype=self.hparams['dtype'])
            F = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT, self.basis_test_OmegaT, theta_q)
            F = F.reshape(((theta_q.shape[0],)+self.hparams['h']))
        if  self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM' or self.hparams['modeltype']=='projection':
            # theta_x0 = discretize_functions(theta_q, self.xi_Gamma_x0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
            # theta_xL = discretize_functions(theta_q, self.xi_Gamma_xL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
            basis_trial = self.basis_trial.forward(self.xi_OmegaT)
            gradbasis_test = self.basis_test.grad(self.xi_OmegaT)[:,:,1:]
            gradbasis_trial = self.basis_trial.grad(self.xi_OmegaT)[:,:,1:]
            ddtbasis_test = self.basis_test.grad(self.xi_OmegaT)[:,:,0]
            basis_test_x0 = self.basis_test.forward(self.xi_Gamma_x0)
            basis_test_xL = self.basis_test.forward(self.xi_Gamma_xL)
            gradbasis_test_x0 = self.basis_test.grad(self.xi_Gamma_x0)[:,:,1:]
            gradbasis_test_xL = self.basis_test.grad(self.xi_Gamma_xL)[:,:,1:]
            basis_trial_x0 = self.basis_trial.forward(self.xi_Gamma_x0)
            basis_trial_xL = self.basis_trial.forward(self.xi_Gamma_xL)
            gradbasis_trial_x0 = self.basis_trial.grad(self.xi_Gamma_x0)[:,:,1:]
            gradbasis_trial_xL = self.basis_trial.grad(self.xi_Gamma_xL)[:,:,1:]
            basis_test_tT = self.basis_test.forward(self.xi_Gamma_tT)
            basis_trial_tT = self.basis_trial.forward(self.xi_Gamma_tT)
            # F = np.zeros((theta.shape[0],self.hparams['N'],self.hparams['N']), dtype=self.hparams['dtype'])
            F = -opt_einsum.contract('q,qn,qm->mn', self.w_OmegaT, basis_trial, ddtbasis_test)
            F += opt_einsum.contract('q,qm,qn->mn', self.w_Gamma_tT, basis_test_tT, basis_trial_tT)   
            F = opt_einsum.contract('q,Nq,qmx,qnx->mn', self.w_OmegaT, theta_q, gradbasis_test, gradbasis_trial)
            F += -opt_einsum.contract('q,qm,x,Nq,qnx->mn', self.w_Gamma_x0, basis_test_x0, self.n_x0, theta_x0_q, gradbasis_trial_x0)
            F += -opt_einsum.contract('q,qm,x,Nq,qnx->mn', self.w_Gamma_xL, basis_test_xL, self.n_xL, theta_xL_q, gradbasis_trial_xL)
            F += -opt_einsum.contract('q,qn,x,Nq,qmx->mn', self.w_Gamma_x0, basis_trial_x0, self.n_x0, theta_x0_q, gradbasis_test_x0)
            F += -opt_einsum.contract('q,qn,x,Nq,qmx->mn', self.w_Gamma_xL, basis_trial_xL, self.n_xL, theta_xL_q, gradbasis_test_xL)         
            if self.hparams.get('gamma_stabilization',0)!=0:
                F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->mn', self.w_Gamma_x0, basis_test_x0, theta_x0_q, basis_trial_x0)
                F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->mn', self.w_Gamma_xL, basis_test_xL, theta_xL_q, basis_trial_xL)
        return F

    def compute_F_tensorized(self, theta_q):
        theta_l = self.project_input_function(theta_q)
        K_c = self.K_t_psiphi_tT - self.K_t_dpsiphi
        K_c = np.kron(K_c, self.K_x_psiphi)
        K_c = np.kron(K_c, self.K_y_psiphi)
        K_lmn = self.K_x_psidpsidphi - self.K_x_psipsidphi_xL + self.K_x_psipsidphi_x0 - self.K_x_psidpsiphi_xL + self.K_x_psidpsiphi_x0
        K_lmn = np.kron(K_lmn, self.K_y_psipsiphi)
        K_lmn = K_lmn + np.kron(self.K_x_psipsiphi, self.K_y_psidpsidphi)
        K_lmn = np.kron(self.K_t_psipsiphi, K_lmn)
        K = opt_einsum.contract('Nl,lmn->Nmn', theta_l, K_lmn) + K_c
        return K

    def compute_F(self, theta_q, theta_x0_q, theta_xL_q):
        if self.hparams['project_materialparameters']==True:
            F = self.compute_F_tensorized(theta_q)
        if self.hparams['project_materialparameters']==False:
            F = self.compute_F_direct(theta_q, theta_x0_q, theta_xL_q)
        return F

    def compute_A_0(self):
        theta_l = np.ones((1,self.hparams['N']))
        K_c = self.K_t_psiphi_tT - self.K_t_dpsiphi
        K_c = np.kron(K_c, self.K_x_psiphi)
        K_c = np.kron(K_c, self.K_y_psiphi)
        K_lmn = self.K_x_psidpsidphi - self.K_x_psipsidphi_xL + self.K_x_psipsidphi_x0 - self.K_x_psidpsiphi_xL + self.K_x_psidpsiphi_x0
        K_lmn = np.kron(K_lmn, self.K_y_psipsiphi)
        K_lmn = K_lmn + np.kron(self.K_x_psipsiphi, self.K_y_psidpsidphi)
        K_lmn = np.kron(self.K_t_psipsiphi, K_lmn)
        F_0 = opt_einsum.contract('Nl,lmn->Nmn', theta_l, K_lmn) + K_c
        A_0 = torch.tensor(np.linalg.inv(F_0), dtype=self.hparams['dtype'], device=self.used_device)
        return A_0
    
    def compute_d_direct(self, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        # f = discretize_functions(f, self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        # eta_y0 = discretize_functions(eta_y0, self.xi_Gamma_y0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        # eta_yL = discretize_functions(eta_yL, self.xi_Gamma_yL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        # g_x0 = discretize_functions(g_x0, self.xi_Gamma_x0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        # g_xL = discretize_functions(g_xL, self.xi_Gamma_xL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        # u0 = discretize_functions(u0, self.xi_Gamma_t0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        # basis_test = self.basis_test.forward(self.xi_OmegaT)
        # basis_test_y0 = self.basis_test.forward(self.xi_Gamma_y0)
        # basis_test_yL = self.basis_test.forward(self.xi_Gamma_yL)
        # basis_test_x0 = self.basis_test.forward(self.xi_Gamma_x0)
        # basis_test_xL = self.basis_test.forward(self.xi_Gamma_xL)
        # gradbasis_test_x0 = self.basis_test.grad(self.xi_Gamma_x0)[:,:,1:]
        # gradbasis_test_xL = self.basis_test.grad(self.xi_Gamma_xL)[:,:,1:]
        # basis_test_t0 = self.basis_test.forward(self.xi_Gamma_t0)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT, self.basis_test_OmegaT, f_q)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_y0, self.basis_test_Gamma_y0, eta_y0_q)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_yL, self.basis_test_Gamma_yL, eta_yL_q)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_x0, self.n_x0, self.gradbasis_test_Gamma_x0, g_x0_q)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_xL, self.n_xL, self.gradbasis_test_Gamma_xL, g_xL_q)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_t0, self.basis_test_Gamma_t0, u0_t0_q)
        if self.hparams.get('gamma_stabilization',0)!=0:
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_x0, self.basis_test_Gamma_x0, g_x0_q)
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_xL, self.basis_test_Gamma_xL, g_xL_q)         
        return d

    # def compute_d_tensorized(self, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
    #     f_l = self.project_input_function(f_q)
    #     eta_l = self.project_input_function(eta_yL_q)
    #     g_l = self.project_input_function(g_xL_q)
    #     u0_l = self.project_input_function(u0_to_q)
    #     M_f = np.kron(np.kron(self.M_t_psi, self.M_x_psi), self.M_y_psi)
    #     M_eta = np.kron(np.kron(self.M_t_psi, self.M_x_psi), self.M_y_psi_y0 + self.M_y_psi_yL)
    #     M_g = np.kron(np.kron(self.M_t_psi, self.M_x_psidpsi_x0 - self.M_x_psidpsi_xL), self.M_y_psi)
    #     M_u0 = np.kron(np.kron(self.M_t_psi_t0, self.M_x_psi), self.M_y_psi)
    #     d = opt_einsum.contract('Nl,lm->Nm', f_l, M_f)
    #     d += opt_einsum.contract('Nl,lm->Nm', eta_l, M_eta)
    #     d += opt_einsum.contract('Nl,lm->Nm', g_l, M_g)
    #     d += opt_einsum.contract('Nl,lm->Nm', u0_l, M_u0)
    #     return d

    def compute_d(self, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        # if self.hparams['project_rhs']==True:
        #     d = self.compute_d_tensorized(f, eta_y0, eta_yL, g_x0, g_xL, u0)
        # else:
        d = self.compute_d_direct(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q)
        return d
    
    #Conserved quantity
    def compute_C(self, f_q, eta_y0_q, eta_yL_q, u0_t0_q):
        # f_d = discretize_functions(f, self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        # eta_y0_d = discretize_functions(eta_y0, self.xi_Gamma_y0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        # eta_yL_d = discretize_functions(eta_yL, self.xi_Gamma_yL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        # u0_d = discretize_functions(u0, self.xi_Gamma_t0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        C = opt_einsum.contract('q,Nq->N', self.w_OmegaT, f_q)
        C += opt_einsum.contract('q,Nq->N', self.w_Gamma_y0, eta_y0_q)
        C += opt_einsum.contract('q,Nq->N', self.w_Gamma_yL, eta_yL_q)
        C += opt_einsum.contract('q,Nq->N', self.w_Gamma_t0, u0_t0_q)
        return C
   
    #Conserved quantity per DOF
    def compute_C_m(self, theta_x0_q, theta_xL_q):
        # theta_x0 = discretize_functions(theta, self.xi_Gamma_x0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        # theta_xL = discretize_functions(theta, self.xi_Gamma_xL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        # basis_trial_tT = self.basis_trial.forward(self.xi_Gamma_tT)
        # gradbasis_trial_x0 = self.basis_trial.grad(self.xi_Gamma_x0)[:,:,1:]
        # gradbasis_trial_xL = self.basis_trial.grad(self.xi_Gamma_xL)[:,:,1:]
        C_m = -opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_x0, self.n_x0, self.gradbasis_trial_Gamma_x0, theta_x0_q)
        C_m += -opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_xL, self.n_xL, self.gradbasis_trial_Gamma_xL, theta_xL_q)
        C_m += opt_einsum.contract('q,qm->m', self.w_Gamma_tT, self.basis_trial_Gamma_tT)
        return C_m
        
    def NN_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        theta_q = theta_q.reshape((theta_q.shape[0],)+(self.hparams['Q']))
        f_q = f_q.reshape((f_q.shape[0],)+(self.hparams['Q']))
        eta_y0_q = eta_y0_q.reshape((eta_y0_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][1],))
        eta_yL_q = eta_yL_q.reshape((eta_yL_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][1],))
        g_x0_q = g_x0_q.reshape((g_x0_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][2],))
        g_xL_q = g_xL_q.reshape((g_xL_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][2],))  
        u0_t0_q = u0_t0_q.reshape((u0_t0_q.shape[0],)+(self.hparams['Q'][1],)+(self.hparams['Q'][2],))       
        eta_q = torch.zeros((eta_y0_q.shape[0],)+(self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        eta_q[:,:,:,0] = eta_y0_q
        eta_q[:,:,:,-1] = eta_yL_q
        g_q = torch.zeros((g_x0_q.shape[0],)+(self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        g_q[:,:,0,:] = g_x0_q
        g_q[:,:,-1,:] = g_xL_q
        u0_t0 = torch.zeros((u0_t0_q.shape[0],)+(self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        u0_t0[:,0,:,:] = u0_t0_q
        u0_t0_q = u0_t0
        inputfuncs = torch.stack((theta_q,f_q,eta_q,g_q,u0_t0_q), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta_q.flatten(-3,-1),f_q.flatten(-3,-1),eta_y0_q,eta_yL_q,g_x0_q,g_xL_q,u0_t0_q), dim=1)
        u_q_hat = self.systemnet.forward(inputfuncs).reshape((theta_q.shape[0],)+(np.prod(self.hparams['Q_L']),))
        return u_q_hat
    
    def NN_simforward(self, theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q, u_q):
        theta_q = torch.tensor(theta_q, dtype=self.hparams['dtype'], device=self.used_device)
        f_q = torch.tensor(f_q, dtype=self.hparams['dtype'], device=self.used_device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.hparams['dtype'], device=self.used_device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.hparams['dtype'], device=self.used_device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.hparams['dtype'], device=self.used_device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.hparams['dtype'], device=self.used_device)   
        u0_t0_q = torch.tensor(u0_t0_q, dtype=self.hparams['dtype'], device=self.used_device)
        if self.hparams['systemnet']==FNO:
            u_q_hat = self.forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q)
        else:
            u_q_hat = self.forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q).detach().cpu().numpy()
        return u_q_hat
    
    def DeepONet_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        theta_q = theta_q.reshape((theta_q.shape[0],)+(self.hparams['Q']))
        f_q = f_q.reshape((f_q.shape[0],)+(self.hparams['Q']))
        eta_y0_q = eta_y0_q.reshape((eta_y0_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][1],))
        eta_yL_q = eta_yL_q.reshape((eta_yL_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][1],))
        g_x0_q = g_x0_q.reshape((g_x0_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][2],))
        g_xL_q = g_xL_q.reshape((g_xL_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][2],))  
        u0_t0_q = u0_t0_q.reshape((u0_t0_q.shape[0],)+(self.hparams['Q'][1],)+(self.hparams['Q'][2],))       
        eta_q = torch.zeros((eta_y0_q.shape[0],)+(self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        eta_q[:,:,:,0] = eta_y0_q
        eta_q[:,:,:,-1] = eta_yL_q
        g_q = torch.zeros((g_x0_q.shape[0],)+(self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        g_q[:,:,0,:] = g_x0_q
        g_q[:,:,-1,:] = g_xL_q
        u0_t0 = torch.zeros((u0_t0_q.shape[0],)+(self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        u0_t0[:,0,:,:] = u0_t0_q
        u0_t0_q = u0_t0
        inputfuncs = torch.stack((theta_q,f_q,eta_q,g_q,u0_t0_q), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta_q.flatten(-3,-1),f_q.flatten(-3,-1),eta_y0_q,eta_yL_q,g_x0_q,g_xL_q,u0_t0_q), dim=1)
        u_m_hat = self.systemnet.forward(inputfuncs).reshape((theta_q.shape[0],self.hparams['N']))
        return u_m_hat
    
    def DeepONet_simforward(self, theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q, u_q):
        theta_q = torch.tensor(theta_q, dtype=self.hparams['dtype'], device=self.used_device)
        f_q = torch.tensor(f_q, dtype=self.hparams['dtype'], device=self.used_device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.hparams['dtype'], device=self.used_device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.hparams['dtype'], device=self.used_device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.hparams['dtype'], device=self.used_device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.hparams['dtype'], device=self.used_device)   
        u0_t0_q = torch.tensor(u0_t0_q, dtype=self.hparams['dtype'], device=self.used_device)  
        u_m_hat = self.forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q).detach().cpu().numpy()
        return u_m_hat
    
    def VarMiON_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        eta_q = torch.zeros((eta_y0_q.shape[0],2*eta_y0_q.shape[1]), dtype=self.hparams['dtype'], device=self.used_device)
        eta_q[:,:eta_y0_q.shape[1]] = eta_y0_q
        eta_q[:,eta_y0_q.shape[1]:] = eta_yL_q
        g_q = torch.zeros((g_x0_q.shape[0],2*g_x0_q.shape[1]), dtype=self.hparams['dtype'], device=self.used_device)
        g_q[:,:g_x0_q.shape[1]] = g_x0_q
        g_q[:,g_x0_q.shape[1]:] = g_xL_q
        systemnet = self.systemnet.forward(theta_q)
        LBranch = self.LBranch_f.forward(f_q) + self.LBranch_eta.forward(eta_q) + self.LBranch_g.forward(g_q) + self.LBranch_u0.forward(u0_t0_q)
        u_m_hat = opt_einsum.contract('nij,nj->ni', systemnet, LBranch)
        return u_m_hat
    
    def VarMiON_simforward(self, theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q, u_q):
        theta_q = torch.tensor(theta_q, dtype=self.hparams['dtype'], device=self.used_device)
        f_q = torch.tensor(f_q, dtype=self.hparams['dtype'], device=self.used_device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.hparams['dtype'], device=self.used_device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.hparams['dtype'], device=self.used_device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.hparams['dtype'], device=self.used_device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.hparams['dtype'], device=self.used_device)   
        u0_t0_q = torch.tensor(u0_t0_q, dtype=self.hparams['dtype'], device=self.used_device)
        u_m_hat = self.forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q).detach().cpu().numpy()
        return u_m_hat

    def NGO_forward(self, F, d, C, C_m):
        if self.hparams.get('scaling_equivariance',False)==True:
            scaling = torch.frobenius_norm(F, dim=(-1,-2))
            # lambdas = torch.linalg.eigvals(F)
            # lambdas_abs = torch.real((lambdas*torch.conj(lambdas))**(1/2))
            # lambda_max = torch.amax(lambdas_abs, axis=-1)
            # scaling = lambda_max
            F = F/scaling[:,None,None]
        if self.hparams.get('Neumannseries', False)==False:
            A_hat = self.systemnet.forward(F)
        if self.hparams.get('Neumannseries', False)==True:
            A_0 = self.A_0*scaling[:,None,None] if self.hparams.get('scaling_equivariance',False)==True else self.A_0
            T = torch.zeros(F.shape, dtype=self.hparams['dtype'], device=self.used_device)
            Ti = self.Identity
            T1 = -F@A_0 + self.Identity
            for i in range(0, self.hparams['Neumannseries_order']):
                Ti = T1@Ti
                T = T + Ti
            A_hat = A_0@(self.Identity + T + self.systemnet.forward(T1))
        if self.hparams.get('scaling_equivariance',False)==True:
            A_hat = A_hat/scaling[:,None,None]        
        u_m_hat = opt_einsum.contract('nij,nj->ni', A_hat, d)
        if self.hparams['massconservation']==True:
            C_hat = opt_einsum.contract('Nn,Nn->N', u_m_hat, C_m)
            l = (C - C_hat)/opt_einsum.contract('Nn,Nn->N', C_m, C_m)
            u_m_hat = u_m_hat + l[:,None]*C_m
        return u_m_hat
    
    def NGO_simforward(self, theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q, u_q):
        F = torch.tensor(self.compute_F(theta_q, theta_x0_q, theta_xL_q), dtype=self.hparams['dtype'], device=self.used_device)
        d = torch.tensor(self.compute_d(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q), dtype=self.hparams['dtype'], device=self.used_device)
        C = torch.tensor(self.compute_C(f_q, eta_y0_q, eta_yL_q, u0_t0_q), dtype=self.hparams['dtype'], device=self.used_device)
        C_m = torch.tensor(self.compute_C_m(theta_x0_q, theta_xL_q), dtype=self.hparams['dtype'], device=self.used_device)
        u_m_hat = self.forward(F, d, C, C_m).detach().cpu().numpy()
        return u_m_hat

    def FEM_forward(self, F, d):
        if self.hparams['Neumannseries']==False:
            K_inv = torch.linalg.pinv(F)
        if self.hparams['Neumannseries']==True:
            T = torch.zeros(F.shape, dtype=self.hparams['dtype'], device=self.used_device)
            Ti = self.Identity
            T1 = -F@self.A_0 + self.Identity
            for i in range(0, self.hparams['Neumannseries_order']):
                Ti = T1@Ti
                T = T + Ti
            K_inv = self.A_0@(self.Identity + T)
        u_m_hat = opt_einsum.contract('nij,nj->ni', K_inv, d)
        return u_m_hat
    
    def FEM_simforward(self, theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q, u_q):
        F = torch.tensor(self.compute_F(theta_q, theta_x0_q, theta_xL_q), dtype=self.hparams['dtype'], device=self.used_device)
        d = torch.tensor(self.compute_d(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q), dtype=self.hparams['dtype'], device=self.used_device)
        u_m_hat = self.forward(F, d).detach().cpu().numpy()
        return u_m_hat
    
    def projection_forward(self, u_q):
        u_m_hat = self.project_input_function(u_q)
        return u_m_hat
    
    def projection_simforward(self, theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q, u_q):
        u_m_hat = self.forward(u_q)
        return u_m_hat
    
    def init_modeltype(self):
        if self.hparams['modeltype']=='NN':
            self.forwardfunction = self.NN_forward
            self.simforwardfunction = self.NN_simforward
        if self.hparams['modeltype']=='DeepONet':
            self.forwardfunction = self.DeepONet_forward
            self.simforwardfunction = self.DeepONet_simforward
        if self.hparams['modeltype']=='VarMiON':
            self.forwardfunction = self.VarMiON_forward
            self.simforwardfunction = self.VarMiON_simforward
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO':
            self.forwardfunction = self.NGO_forward
            self.simforwardfunction = self.NGO_simforward
        if self.hparams['modeltype']=='FEM':
            self.forwardfunction = self.FEM_forward
            self.simforwardfunction = self.FEM_simforward
        if self.hparams['modeltype']=='projection':
            self.forwardfunction = self.projection_forward
            self.simforwardfunction = self.projection_simforward
    
    def forward(self, *args):
        if self.hparams['modeltype']=='NN':
            u_q_hat = self.forwardfunction(*args)
            return u_q_hat
        if self.hparams['modeltype']!='NN':
            u_m = self.forwardfunction(*args)
            if self.hparams['output_coefficients']==True:
                return u_m
            if self.hparams['output_coefficients']==False:
                u_hat = opt_einsum.contract('Nn,qn->Nq', u_m, self.psix)
                return u_hat
    
    def simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, x, u):
            u_q_hat = np.zeros((len(theta),len(x)))
            self.compute_quadrature()
            # self.compute_basis_evaluations()
            # self.rescale_quadrature_t()
            x_scaled = np.copy(x)
            x_scaled[:,0] = x_scaled[:,0]*self.hparams['Dt']
            # self.compute_Kronecker_factors_t()
            # if self.hparams['Neumannseries']==True:
            #     self.A_0 = self.compute_A_0()
            for i in range(self.hparams['n_timesteps']):
                print(i)
                i_t = np.where((x[:,0]>=i*self.hparams['Dt'])&(x[:,0]<=(i+1)*self.hparams['Dt']))[0]
                u_q = discretize_functions(u, self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.hparams['discretization_device']) 
                theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
                if i>0:
                    u0_t0_q = opt_einsum.contract('Nn,qn->Nq', u_m_hat, self.basis_trial_Gamma_tT)
                if self.hparams['modeltype']=='NN':
                    theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
                    u_q_hat = self.simforwardfunction(theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q, u_q)
                else:
                    u_m_hat = self.simforwardfunction(theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q, u_q)
                    # u_q_hat_Dt = opt_einsum.contract('Nn,qn->Nq', u_m_hat, self.basis_trial.forward(x))
                    # u_q_exact = discretize_functions(u, x_scaled, dtype=self.hparams['dtype'], device=self.hparams['discretization_device']) 
                    # print('Dt error: '+ str(np.average(weightedrelativeL2_set(self.w_OmegaT, u_q_hat_Dt, u_q_exact))))
                    # print('Dt error: '+ str(np.average(weightedrelativeL2_set(self.w_OmegaT, u_q_hat_Dt, u_q))))
                    # Collect solution
                    x_Dt_scaled = x[i_t]
                    x_Dt_scaled[:,0] = (x_Dt_scaled[:,0] - i*self.hparams['Dt'])/self.hparams['Dt']
                    u_q_hat[:,i_t] = opt_einsum.contract('Nn,qn->Nq', u_m_hat, self.basis_trial.forward(x_Dt_scaled))
                    # u_q_hat[:,i_t] = opt_einsum.contract('Nn,qn->Nq', u_m_hat, self.basis_trial.forward(x))
                # self.translate_quadrature_t()
                x_scaled[:,0] = x_scaled[:,0] + self.hparams['Dt']    
            return u_q_hat

    def compute_quadrature(self):
        #Quadrature
        if self.hparams['quadrature']=='Gauss-Legendre':
            quad_OmegaT = GaussLegendreQuadrature(Q=self.hparams['Q'], n_elements=self.hparams['n_elements'])
            quad_Gamma_t = GaussLegendreQuadrature(Q=[self.hparams['Q'][1],self.hparams['Q'][2]], n_elements=[self.hparams['n_elements'][1],self.hparams['n_elements'][2]])
            quad_Gamma_x = GaussLegendreQuadrature(Q=[self.hparams['Q'][0],self.hparams['Q'][2]], n_elements=[self.hparams['n_elements'][0],self.hparams['n_elements'][2]])
            quad_Gamma_y = GaussLegendreQuadrature(Q=[self.hparams['Q'][0],self.hparams['Q'][1]], n_elements=[self.hparams['n_elements'][0],self.hparams['n_elements'][1]])
            quad_t = GaussLegendreQuadrature(Q=[self.hparams['Q'][0]], n_elements=[self.hparams['n_elements'][0]])
            quad_x = GaussLegendreQuadrature(Q=[self.hparams['Q'][1]], n_elements=[self.hparams['n_elements'][1]])
            quad_y = GaussLegendreQuadrature(Q=[self.hparams['Q'][2]], n_elements=[self.hparams['n_elements'][2]])

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
            self.xi_Gamma_xL[:,0] = quad_Gamma_x.xi[:,0]
            self.xi_Gamma_xL[:,2] = quad_Gamma_x.xi[:,1]

            self.w_Gamma_y0 = quad_Gamma_y.w
            self.xi_Gamma_y0 = np.zeros((self.hparams['Q'][0]*self.hparams['Q'][1],self.hparams['d']))
            self.xi_Gamma_y0[:,0] = quad_Gamma_y.xi[:,0]
            self.xi_Gamma_y0[:,1] = quad_Gamma_y.xi[:,1]

            self.w_Gamma_yL = quad_Gamma_y.w
            self.xi_Gamma_yL = np.ones((self.hparams['Q'][0]*self.hparams['Q'][1],self.hparams['d']))
            self.xi_Gamma_yL[:,0] = quad_Gamma_y.xi[:,0]
            self.xi_Gamma_yL[:,1] = quad_Gamma_y.xi[:,1]

            self.w_t = quad_t.w
            self.xi_t = quad_t.xi[:,0]
            self.w_x = quad_x.w
            self.xi_x = quad_x.xi[:,0]
            self.w_y = quad_y.w
            self.xi_y = quad_y.xi[:,0]

        #Outward normal
        self.n_y0 = np.array([0,-1])
        self.n_yL = np.array([0,1])
        self.n_x0 = np.array([-1,0])
        self.n_xL = np.array([1,0])

        #Loss quadrature
        if self.hparams['quadrature_L']=='Gauss-Legendre':
            quad_OmegaT_L = GaussLegendreQuadrature(Q=self.hparams['Q_L'], n_elements=self.hparams['n_elements_L'])
        self.w_OmegaT_L = quad_OmegaT_L.w
        self.xi_OmegaT_L = quad_OmegaT_L.xi

    def rescale_quadrature_t(self):
        self.w_OmegaT = self.w_OmegaT*self.hparams['Dt']
        self.w_Gamma_x0 = self.w_Gamma_x0*self.hparams['Dt']
        self.w_Gamma_xL = self.w_Gamma_xL*self.hparams['Dt']
        self.w_Gamma_y0 = self.w_Gamma_y0*self.hparams['Dt']
        self.w_Gamma_yL = self.w_Gamma_yL*self.hparams['Dt']
        self.w_t = self.w_t*self.hparams['Dt']
        self.gradbasis_test_t = 1/self.hparams['Dt']*self.gradbasis_test_t

        self.xi_OmegaT[:,0] = self.xi_OmegaT[:,0]*self.hparams['Dt']
        self.xi_Gamma_t0[:,0] = self.xi_Gamma_t0[:,0]*self.hparams['Dt']
        self.xi_Gamma_tT[:,0] = self.xi_Gamma_tT[:,0]*self.hparams['Dt']
        self.xi_Gamma_x0[:,0] = self.xi_Gamma_x0[:,0]*self.hparams['Dt']
        self.xi_Gamma_xL[:,0] = self.xi_Gamma_xL[:,0]*self.hparams['Dt']
        self.xi_Gamma_y0[:,0] = self.xi_Gamma_y0[:,0]*self.hparams['Dt']
        self.xi_Gamma_yL[:,0] = self.xi_Gamma_yL[:,0]*self.hparams['Dt']

        self.xi_t = self.xi_t*self.hparams['Dt']

    def translate_quadrature_t(self):
        self.xi_OmegaT[:,0] += self.hparams['Dt']
        self.xi_Gamma_t0[:,0] += self.hparams['Dt']
        self.xi_Gamma_tT[:,0] += self.hparams['Dt']
        self.xi_Gamma_x0[:,0] += self.hparams['Dt']
        self.xi_Gamma_xL[:,0] += self.hparams['Dt']
        self.xi_Gamma_y0[:,0] += self.hparams['Dt']
        self.xi_Gamma_yL[:,0] += self.hparams['Dt']
        self.xi_t += self.hparams['Dt']

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
                if self.hparams['output_coefficients']==True:
                    loss += self.hparams['solution_loss'](self.M_phiphi, u_hat, u)
                if self.hparams['output_coefficients']==False:
                    loss += self.hparams['solution_loss'](self.w_OmegaT_L, u_hat, u)
            if self.hparams.get('matrix_loss',None)!=None:
                F = inputs[0]
                if self.hparams.get('Neumannseries', False)==False:
                    A_hat = self.systemnet.forward(F)
                if self.hparams.get('Neumannseries', False)==True:
                    A_0 = self.A_0 if self.hparams.get('A0net')==None else self.A0net.systemnet.forward(F)
                    T = torch.zeros(F.shape, dtype=self.hparams['dtype'], device=self.used_device)
                    Ti = self.Identity
                    T1 = -F@A_0 + self.Identity
                    for i in range(0, self.hparams['Neumannseries_order']):
                        Ti = T1@Ti
                        T = T + Ti
                    A_hat = A_0@(self.Identity + T + self.systemnet.forward(T1))
                loss += self.hparams['matrix_loss'](torch.matmul(F, torch.matmul(A_hat,F)), F)
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
            if self.hparams['output_coefficients']==True:
                loss += self.hparams['solution_loss'](self.M_phiphi, u_hat, u)
            if self.hparams['output_coefficients']==False:
                loss += self.hparams['solution_loss'](self.w_OmegaT_L, u_hat, u)
        if self.hparams.get('matrix_loss',None)!=None:
            F = inputs[0]
            if self.hparams.get('Neumannseries',False)==False:
                A_hat = self.systemnet.forward(F)
            if self.hparams.get('Neumannseries', False)==True:
                A_0 = self.A_0 if self.hparams.get('A0net')==None else self.A0net.systemnet.forward(F)
                T = torch.zeros(F.shape, dtype=self.hparams['dtype'], device=self.used_device)
                Ti = self.Identity
                T1 = -F@A_0 + self.Identity
                for i in range(0, self.hparams['Neumannseries_order']):
                    Ti = T1@Ti
                    T = T + Ti
                A_hat = A_0@(self.Identity + T + self.systemnet.forward(T1))
            loss += self.hparams['matrix_loss'](torch.matmul(F, torch.matmul(A_hat,F)), F)
            print(loss)
        if self.hparams['output_coefficients']==True:
            metric = self.hparams['metric'](self.M_phiphi, u_hat, u)
        if self.hparams['output_coefficients']==False:
            metric = self.hparams['metric'](self.w_OmegaT_L, u_hat, u)
        self.metric.append(metric)
        self.log('val_loss', loss)
        self.log('metric', metric)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['hparams'] = self.hparams
        
    def on_fit_start(self):
        self.used_device = self.device
        torch.set_num_threads(2)
        if self.hparams['modeltype']!='NN':
            if self.hparams['output_coefficients']==False:
                self.psix = torch.tensor(self.basis_trial.forward(self.xi_OmegaT_L), dtype=self.hparams['dtype'], device=self.used_device)
                self.psix = self.psix.to(self.used_device)
            if self.hparams['output_coefficients']==True:
                self.M_phiphi = torch.tensor(self.M_phiphi, dtype=self.hparams['dtype'], device=self.used_device).to(self.used_device)
        self.w_OmegaT_L = torch.tensor(self.w_OmegaT_L, dtype=self.hparams['dtype'], device=self.used_device).to(self.used_device)
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