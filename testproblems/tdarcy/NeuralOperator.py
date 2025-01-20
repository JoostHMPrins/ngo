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
        self.phix = torch.tensor(self.basis_trial.forward(self.xi_OmegaT_L), dtype=self.hparams['dtype'], device=self.used_device)
        #Relevant matrices
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM':
            self.Identity = torch.eye(self.hparams['N'], dtype=self.hparams['dtype'], device=self.used_device)
            self.compute_Kronecker_factors()
            if self.hparams['Neumannseries']==True:
                self.A_0 = self.compute_A_0()

    def discretize_input_functions(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        theta_d = discretize_functions(theta, self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        theta_x0_d = discretize_functions(theta, self.xi_Gamma_x0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        theta_xL_d = discretize_functions(theta, self.xi_Gamma_xL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        f_d = discretize_functions(f, self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        eta_y0_d = discretize_functions(eta_y0, self.xi_Gamma_y0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        eta_yL_d = discretize_functions(eta_yL, self.xi_Gamma_yL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        g_x0_d = discretize_functions(g_x0, self.xi_Gamma_x0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        g_xL_d = discretize_functions(g_xL, self.xi_Gamma_xL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        u0_d = discretize_functions(u0, self.xi_Gamma_t0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        return theta_d, theta_x0_d, theta_xL_d, f_d, eta_y0_d, eta_yL_d, g_x0_d, g_xL_d, u0_d
    
    def discretize_output_function(self, u):
        u_d = discretize_functions(u, self.xi_OmegaT_L, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        return u_d
    
    def compute_projection_coeffs(self, u):
        basis_test = self.basis_test.forward(self.xi_OmegaT)
        u_q = discretize_functions(u, self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT, basis_test, u_q)
        M_inv = np.kron(np.kron(self.M_t_phipsi_inv, self.M_x_phipsi_inv), self.M_y_phipsi_inv)
        u_n = opt_einsum.contract('mn,Nm->Nn', M_inv, d)
        return u_n

    def compute_Kronecker_factors(self):
        zero = np.zeros((1,1))
        one = np.ones((1,1))
        basis_test_t = self.hparams['test_bases'][0].forward(self.xi_t)
        basis_test_x = self.hparams['test_bases'][1].forward(self.xi_x)
        basis_test_y = self.hparams['test_bases'][2].forward(self.xi_y)
        gradbasis_test_t = self.hparams['test_bases'][0].grad(self.xi_t)
        gradbasis_test_x = self.hparams['test_bases'][1].grad(self.xi_x)
        gradbasis_test_y = self.hparams['test_bases'][2].grad(self.xi_y)   
        basis_trial_t = self.hparams['trial_bases'][0].forward(self.xi_t)
        basis_trial_x = self.hparams['trial_bases'][1].forward(self.xi_x)
        basis_trial_y = self.hparams['trial_bases'][2].forward(self.xi_y)
        # gradbasis_trial_t = self.hparams['trial_bases'][0].grad(self.xi_t), dtype=self.hparams['dtype'], device=self.used_device)
        gradbasis_trial_x = self.hparams['trial_bases'][1].grad(self.xi_x)
        gradbasis_trial_y = self.hparams['trial_bases'][2].grad(self.xi_y)
        basis_test_t0 = self.hparams['test_bases'][0].forward(zero)[0]
        basis_test_x0 = self.hparams['test_bases'][1].forward(zero)[0]
        basis_test_y0 = self.hparams['test_bases'][2].forward(zero)[0]
        # gradbasis_test_t0 = self.hparams['test_bases'][0].grad(zero)[0], dtype=self.hparams['dtype'], device=self.used_device)
        gradbasis_test_x0 = self.hparams['test_bases'][1].grad(zero)[0]
        # gradbasis_test_y0 = self.hparams['test_bases'][2].grad(zero)[0], dtype=self.hparams['dtype'], device=self.used_device)
        # basis_trial_t0 = self.hparams['trial_bases'][0].forward(zero)[0], dtype=self.hparams['dtype'], device=self.used_device)
        basis_trial_x0 = self.hparams['trial_bases'][1].forward(zero)[0]
        # basis_trial_y0 = self.hparams['trial_bases'][2].forward(zero)[0], dtype=self.hparams['dtype'], device=self.used_device)
        # gradbasis_trial_t0 = self.hparams['trial_bases'][0].grad(zero)[0], dtype=self.hparams['dtype'], device=self.used_device)
        gradbasis_trial_x0 = self.hparams['trial_bases'][1].grad(zero)[0]
        # gradbasis_trial_y0 = self.hparams['trial_bases'][2].grad(zero)[0], dtype=self.hparams['dtype'], device=self.used_device)
        basis_test_tT = self.hparams['test_bases'][0].forward(one)[0]
        basis_test_xL = self.hparams['test_bases'][1].forward(one)[0]
        basis_test_yL = self.hparams['test_bases'][2].forward(one)[0]
        # gradbasis_test_tT = self.hparams['test_bases'][0].grad(one)[0], dtype=self.hparams['dtype'], device=self.used_device)
        gradbasis_test_xL = self.hparams['test_bases'][1].grad(one)[0]
        # gradbasis_test_yL = self.hparams['test_bases'][2].grad(one)[0], dtype=self.hparams['dtype'], device=self.used_device)
        basis_trial_tT = self.hparams['trial_bases'][0].forward(one)[0]
        basis_trial_xL = self.hparams['trial_bases'][1].forward(one)[0]
        # basis_trial_yL = self.hparams['trial_bases'][2].forward(one)[0], dtype=self.hparams['dtype'], device=self.used_device)
        # gradbasis_trial_tT = self.hparams['trial_bases'][0].grad(one)[0], dtype=self.hparams['dtype'], device=self.used_device)
        gradbasis_trial_xL = self.hparams['trial_bases'][1].grad(one)[0]
        # gradbasis_trial_yL = self.hparams['trial_bases'][2].grad(one)[0], dtype=self.hparams['dtype'], device=self.used_device)
        self.K_t_phiphipsi = opt_einsum.contract('q,qa,qd,qg->adg', self.w_t, basis_test_t, basis_test_t, basis_trial_t)
        self.K_x_phiphipsi = opt_einsum.contract('q,qb,qe,qh->beh', self.w_x, basis_test_x, basis_test_x, basis_trial_x)
        self.K_y_phiphipsi = opt_einsum.contract('q,qc,qf,qi->cfi', self.w_y, basis_test_y, basis_test_y, basis_trial_y)
        self.K_x_phidphidpsi = opt_einsum.contract('q,qb,qe,qh->beh', self.w_x, basis_test_x, gradbasis_test_x, gradbasis_trial_x)
        self.K_y_phidphidpsi = opt_einsum.contract('q,qc,qf,qi->cfi', self.w_y, basis_test_y, gradbasis_test_y, gradbasis_trial_y)
        self.K_x_phiphidpsi_x0 = opt_einsum.contract('b,e,h->beh', basis_test_x0, basis_test_x0, gradbasis_trial_x0)
        self.K_x_phidphipsi_x0 = opt_einsum.contract('b,e,h->beh', basis_test_x0, gradbasis_test_x0, basis_trial_x0)
        self.K_x_phiphidpsi_xL = opt_einsum.contract('b,e,h->beh', basis_test_xL, basis_test_xL, gradbasis_trial_xL)
        self.K_x_phidphipsi_xL = opt_einsum.contract('b,e,h->beh', basis_test_xL, gradbasis_test_xL, basis_trial_xL)
        self.K_x_phipsi = opt_einsum.contract('q,qe,qh->eh', self.w_x, basis_test_x, basis_trial_x)
        self.K_y_phipsi = opt_einsum.contract('q,qf,qi->fi', self.w_y, basis_test_y, basis_trial_y)
        self.K_t_dphipsi = opt_einsum.contract('q,qd,qg->dg', self.w_t, gradbasis_test_t, basis_trial_t)
        self.K_t_phipsi_tT = opt_einsum.contract('d,g->dg', basis_test_tT, basis_trial_tT)
        self.M_t_phiphi = opt_einsum.contract('q,qa,qd->ad', self.w_t, basis_test_t, basis_test_t)
        self.M_x_phiphi = opt_einsum.contract('q,qb,qe->be', self.w_x, basis_test_x, basis_test_x)
        self.M_y_phiphi = opt_einsum.contract('q,qc,qf->cf', self.w_y, basis_test_y, basis_test_y)
        self.M_y_phiphi_y0 = opt_einsum.contract('c,f->cf', basis_test_y0, basis_test_y0)
        self.M_y_phiphi_yL = opt_einsum.contract('c,f->cf', basis_test_yL, basis_test_yL)
        self.M_x_phidphi = opt_einsum.contract('q,qb,qe->be', self.w_x, basis_test_x, gradbasis_test_x)
        self.M_x_phidphi_x0 = opt_einsum.contract('b,e->be', basis_test_x0, gradbasis_test_x0)
        self.M_x_phidphi_xL = opt_einsum.contract('b,e->be', basis_test_xL, gradbasis_test_xL)
        self.M_t_phiphi_t0 = opt_einsum.contract('a,d->ad', basis_test_t0, basis_test_t0)

        self.M_t_phipsi = opt_einsum.contract('q,qa,qg->ag', self.w_t, basis_test_t, basis_trial_t)
        self.M_x_phipsi = opt_einsum.contract('q,qb,qh->bh', self.w_x, basis_test_x, basis_trial_x)
        self.M_y_phipsi = opt_einsum.contract('q,qc,qi->ci', self.w_y, basis_test_y, basis_trial_y)
        self.M_t_phipsi_inv = np.linalg.pinv(self.M_t_phipsi)
        self.M_x_phipsi_inv = np.linalg.pinv(self.M_x_phipsi)
        self.M_y_phipsi_inv = np.linalg.pinv(self.M_y_phipsi)

    def compute_F_direct(self, theta):
        theta = discretize_functions(theta, self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        if  self.hparams['modeltype']=='data NGO':
            basis_test = torch.tensor(self.basis_test.forward(self.xi_OmegaT), dtype=self.hparams['dtype'])
            F = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT, basis_test, theta)
            F = F.reshape((theta.shape[0],self.hparams['h'][0],self.hparams['h'][1],self.hparams['h'][2]))
        if  self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM' or self.hparams['modeltype']=='projection':
            theta_x0 = discretize_functions(theta, self.xi_Gamma_x0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
            theta_xL = discretize_functions(theta, self.xi_Gamma_xL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
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
            F = opt_einsum.contract('q,Nq,qmx,qnx->mn', self.w_OmegaT, theta, gradbasis_test, gradbasis_trial)
            F += -opt_einsum.contract('q,qm,x,Nq,qnx->mn', self.w_Gamma_x0, basis_test_x0, self.n_x0, theta_x0, gradbasis_trial_x0)
            F += -opt_einsum.contract('q,qm,x,Nq,qnx->mn', self.w_Gamma_xL, basis_test_xL, self.n_xL, theta_xL, gradbasis_trial_xL)
            F += -opt_einsum.contract('q,qn,x,Nq,qmx->mn', self.w_Gamma_x0, basis_trial_x0, self.n_x0, theta_x0, gradbasis_test_x0)
            F += -opt_einsum.contract('q,qn,x,Nq,qmx->mn', self.w_Gamma_xL, basis_trial_xL, self.n_xL, theta_xL, gradbasis_test_xL)         
            if self.hparams.get('gamma_stabilization',0)!=0:
                F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->mn', self.w_Gamma_x0, basis_test_x0, theta_x0, basis_trial_x0)
                F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->mn', self.w_Gamma_xL, basis_test_xL, theta_xL, basis_trial_xL)
        return F

    def compute_F_tensorized(self, theta):
        theta_l = self.compute_projection_coeffs(theta)
        K_c = self.K_t_phipsi_tT - self.K_t_dphipsi
        K_c = np.kron(K_c, self.K_x_phipsi)
        K_c = np.kron(K_c, self.K_y_phipsi)
        K_lmn = self.K_x_phidphidpsi - self.K_x_phiphidpsi_xL + self.K_x_phiphidpsi_x0 - self.K_x_phidphipsi_xL + self.K_x_phidphipsi_x0
        K_lmn = np.kron(K_lmn, self.K_y_phiphipsi)
        K_lmn = K_lmn + np.kron(self.K_x_phiphipsi, self.K_y_phidphidpsi)
        K_lmn = np.kron(self.K_t_phiphipsi, K_lmn)
        K = opt_einsum.contract('Nl,lmn->Nmn', theta_l, K_lmn) + K_c
        return K

    def compute_F(self, theta):
        # if self.hparams['modeltype']=='data NGO':
        #     F = torch.zeros((len(theta),self.hparams['h'][0],self.hparams['h'][1],self.hparams['h'][2]))
        # if self.hparams['modeltype']=='model NGO':
        #     F = torch.zeros((len(theta),self.hparams['N'],self.hparams['N']))
        # for i in range(int(len(theta)/self.hparams['assembly_batch_size'])):
        #     i0 = self.hparams['assembly_batch_size']*i
        #     i1 = self.hparams['assembly_batch_size']*(i+1)
        #     if self.hparams['project_materialparameters']==True:
        #         F[i0:i1] = self.compute_F_tensorized(theta[i0:i1])
        #     if self.hparams['project_materialparameters']==False:
        #         F[i0:i1] = self.compute_F_direct(theta[i0:i1])
        if self.hparams['project_materialparameters']==True:
            F = self.compute_F_tensorized(theta)
        if self.hparams['project_materialparameters']==False:
            F = self.compute_F_direct(theta)
        return F

    def compute_A_0(self):
        theta_l = np.ones((1,self.hparams['N']))
        K_c = self.K_t_phipsi_tT - self.K_t_dphipsi
        K_c = np.kron(K_c, self.K_x_phipsi)
        K_c = np.kron(K_c, self.K_y_phipsi)
        K_lmn = self.K_x_phidphidpsi - self.K_x_phiphidpsi_xL + self.K_x_phiphidpsi_x0 - self.K_x_phidphipsi_xL + self.K_x_phidphipsi_x0
        K_lmn = np.kron(K_lmn, self.K_y_phiphipsi)
        K_lmn = K_lmn + np.kron(self.K_x_phiphipsi, self.K_y_phidphidpsi)
        K_lmn = np.kron(self.K_t_phiphipsi, K_lmn)
        F_0 = opt_einsum.contract('Nl,lmn->Nmn', theta_l, K_lmn) + K_c
        A_0 = torch.tensor(np.linalg.inv(F_0), dtype=self.hparams['dtype'], device=self.used_device)
        return A_0
    
    def compute_d_direct(self, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        f = discretize_functions(f, self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        eta_y0 = discretize_functions(eta_y0, self.xi_Gamma_y0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        eta_yL = discretize_functions(eta_yL, self.xi_Gamma_yL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        g_x0 = discretize_functions(g_x0, self.xi_Gamma_x0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        g_xL = discretize_functions(g_xL, self.xi_Gamma_xL, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        u0 = discretize_functions(u0, self.xi_Gamma_t0, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        basis_test = self.basis_test.forward(self.xi_OmegaT)
        basis_test_y0 = self.basis_test.forward(self.xi_Gamma_y0)
        basis_test_yL = self.basis_test.forward(self.xi_Gamma_yL)
        basis_test_x0 = self.basis_test.forward(self.xi_Gamma_x0)
        basis_test_xL = self.basis_test.forward(self.xi_Gamma_xL)
        gradbasis_test_x0 = self.basis_test.grad(self.xi_Gamma_x0)[:,:,1:]
        gradbasis_test_xL = self.basis_test.grad(self.xi_Gamma_xL)[:,:,1:]
        basis_test_t0 = self.basis_test.forward(self.xi_Gamma_t0)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT, basis_test, f)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_y0, basis_test_y0, eta_y0)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_yL, basis_test_yL, eta_yL)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_x0, self.n_x0, gradbasis_test_x0, g_x0)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_xL, self.n_xL, gradbasis_test_xL, g_xL)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_t0, basis_test_t0, u0)
        if self.hparams.get('gamma_stabilization',0)!=0:
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_x0, basis_test_x0, g_x0)
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_xL, basis_test_xL, g_xL)         
        return d

    def compute_d_tensorized(self, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        f_l = self.compute_projection_coeffs(f)
        eta_l = self.compute_projection_coeffs(eta_yL)
        g_l = self.compute_projection_coeffs(g_xL)
        u0_l = self.compute_projection_coeffs(u0)
        M_f = np.kron(np.kron(self.M_t_phiphi, self.M_x_phiphi), self.M_y_phiphi)
        M_eta = np.kron(np.kron(self.M_t_phiphi, self.M_x_phiphi), self.M_y_phiphi_y0 + self.M_y_phiphi_yL)
        M_g = np.kron(np.kron(self.M_t_phiphi, self.M_x_phidphi_x0 - self.M_x_phidphi_xL), self.M_y_phiphi)
        M_u0 = np.kron(np.kron(self.M_t_phiphi_t0, self.M_x_phiphi), self.M_y_phiphi)
        d = opt_einsum.contract('Nl,lm->Nm', f_l, M_f)
        d += opt_einsum.contract('Nl,lm->Nm', eta_l, M_eta)
        d += opt_einsum.contract('Nl,lm->Nm', g_l, M_g)
        d += opt_einsum.contract('Nl,lm->Nm', u0_l, M_u0)
        return d

    def compute_d(self, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        # d = torch.zeros((len(f),self.hparams['N']))
        # for i in range(int(len(f)/self.hparams['assembly_batch_size'])):
        #     i0 = self.hparams['assembly_batch_size']*i
        #     i1 = self.hparams['assembly_batch_size']*(i+1)
            # if self.hparams['project_rhs']==True:
            #     d[i0:i1] = self.compute_d_tensorized(f[i0:i1], eta_y0[i0:i1], eta_yL[i0:i1], g_x0[i0:i1], g_xL[i0:i1], u0[i0:i1])
            # else:
            #     d[i0:i1] = self.compute_d_direct(f[i0:i1], eta_y0[i0:i1], eta_yL[i0:i1], g_x0[i0:i1], g_xL[i0:i1], u0[i0:i1])
        if self.hparams['project_rhs']==True:
            d = self.compute_d_tensorized(f, eta_y0, eta_yL, g_x0, g_xL, u0)
        else:
            d = self.compute_d_direct(f, eta_y0, eta_yL, g_x0, g_xL, u0)
        return d

    # def NN_forward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0):
    #     theta = theta.reshape((theta.shape[0],self.hparams['Q'],self.hparams['Q']))
    #     f = f.reshape((f.shape[0],self.hparams['Q'],self.hparams['Q']))
    #     eta = torch.zeros((eta_y0.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
    #     eta[:,:,0] = eta_y0
    #     eta[:,:,-1] = eta_yL
    #     g = torch.zeros((g_x0.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
    #     g[:,0,:] = g_x0
    #     g[:,-1,:] = g_xL
    #     inputfuncs = torch.stack((theta,f,eta,g), dim=1)
    #     if self.hparams['systemnet']==MLP:
    #         inputfuncs = torch.cat((theta.flatten(-2,-1),f.flatten(-2,-1),eta_y0,eta_yL,g_x0,g_xL), dim=1)
    #     u_hat = self.systemnet.forward(inputfuncs).reshape((theta.shape[0],self.hparams['Q_L']**self.hparams['d']))
    #     return u_hat
    
    # def DeepONet_forward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0):
    #     theta = theta.reshape((theta.shape[0],self.hparams['Q'],self.hparams['Q']))
    #     f = f.reshape((f.shape[0],self.hparams['Q'],self.hparams['Q']))
    #     eta = torch.zeros((eta_y0.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
    #     eta[:,:,0] = eta_y0
    #     eta[:,:,-1] = eta_yL
    #     g = torch.zeros((g_x0.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
    #     g[:,0,:] = g_x0
    #     g[:,-1,:] = g_xL
    #     inputfuncs = torch.stack((theta,f,eta,g), dim=1)
    #     if self.hparams['systemnet']==MLP:
    #         inputfuncs = torch.cat((theta.flatten(-2,-1),f.flatten(-2,-1),eta_y0,eta_yL,g_x0,g_xL), dim=1)
    #     u_n = self.systemnet.forward(inputfuncs).reshape((theta.shape[0],self.hparams['N']))
    #     u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.phix)
    #     return u_hat
    
    # def VarMiON_forward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0):
    #     eta = torch.zeros((eta_y0.shape[0],2*eta_y0.shape[1]), dtype=self.hparams['dtype'], device=self.used_device)
    #     eta[:,:eta_y0.shape[1]] = eta_y0
    #     eta[:,eta_y0.shape[1]:] = eta_yL
    #     g = torch.zeros((g_x0.shape[0],2*g_x0.shape[1]), dtype=self.hparams['dtype'], device=self.used_device)
    #     g[:,:g_x0.shape[1]] = g_x0
    #     g[:,g_x0.shape[1]:] = g_xL
    #     systemnet = self.systemnet.forward(theta)
    #     LBranch = self.LBranch_f.forward(f) + self.LBranch_eta.forward(eta) + self.LBranch_g.forward(g)
    #     u_n = opt_einsum.contract('nij,nj->ni', systemnet, LBranch)
    #     u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.phix)
    #     return u_hat

    def NGO_forward(self, F, d):
        if self.hparams.get('scaling_equivariance',False)==True:
            F_bar = torch.norm(F, dim=(-1,-2), p='fro')
            F = F/F_bar[:,None,None]
        if self.hparams.get('Neumannseries', False)==False:
            A = self.systemnet.forward(F)
        if self.hparams.get('Neumannseries', False)==True:
            A_0 = self.A_0 if self.hparams.get('A0net')==None else self.A0net.systemnet.forward(F)
            T = torch.zeros(F.shape, dtype=self.hparams['dtype'], device=self.used_device)
            Ti = self.Identity
            T1 = -F@A_0 + self.Identity
            # eigvals = torch.linalg.eigvals(T1)
            # spectralradius = torch.amax(torch.real((eigvals*torch.conj(eigvals))**(1/2)), dim=-1)
            # print('spectralradius: '+str(spectralradius))
            for i in range(0, self.hparams['Neumannseries_order']):
                Ti = T1@Ti
                T = T + Ti
            A = A_0@(self.Identity + T + self.systemnet.forward(T1))
        if self.hparams.get('scaling_equivariance',False)==True:
            A = A/F_bar[:,None,None]        
        u_n = opt_einsum.contract('nij,nj->ni', A, d)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.phix)
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
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.phix)
        return u_hat
    
    def projection_forward(self, u):
        # u_q = discretize_functions(u, self.xi_OmegaT, dtype=self.hparams['dtype'], device=self.hparams['discretization_device'])
        # basis_test = self.basis_test.forward(self.xi_OmegaT)
        # d = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT, basis_test, u_q)
        # M_inv = np.kron(np.kron(self.M_t_phipsi_inv, self.M_x_phipsi_inv), self.M_y_phipsi_inv)
        # u_n = opt_einsum.contract('mn,Nm->Nn', M_inv, d)
        u_n = self.compute_projection_coeffs(u)
        # u_n = torch.tensor(self.compute_projection_coeffs(u), dtype=self.hparams['dtype'], device=self.used_device)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.phix)
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
    
    def simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, x, u):
        self.geometry()
        if self.hparams['modeltype']=='NN' or self.hparams['modeltype']=='DeepONet' or self.hparams['modeltype']=='VarMiON':
            theta_d, theta_x0_d, theta_xL_d, f_d, eta_y0_d, eta_yL_d, g_x0_d, g_xL_d, u0_d = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
            theta_d = torch.tensor(theta_d, dtype=self.hparams['dtype'], device=self.used_device)
            f_d = torch.tensor(f_d, dtype=self.hparams['dtype'], device=self.used_device)
            eta_y0_d = torch.tensor(eta_y0_d, dtype=self.hparams['dtype'], device=self.used_device)
            eta_yL_d = torch.tensor(eta_yL_d, dtype=self.hparams['dtype'], device=self.used_device)
            g_x0_d = torch.tensor(g_x0_d, dtype=self.hparams['dtype'], device=self.used_device)
            g_xL_d = torch.tensor(g_xL_d, dtype=self.hparams['dtype'], device=self.used_device)    
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO' or self.hparams['modeltype']=='FEM':
            F = torch.tensor(self.compute_F(theta), dtype=self.hparams['dtype'], device=self.used_device)
            d = torch.tensor(self.compute_d(f, eta_y0, eta_yL, g_x0, g_xL, u0), dtype=self.hparams['dtype'], device=self.used_device)
        # self.phix = torch.tensor(self.basis_trial.forward(x), dtype=self.hparams['dtype'], device=self.used_device)
        self.phix = self.basis_trial.forward(x)
        if self.hparams['modeltype']=='NN':
            u_hat = self.NN_forward(theta_d, f_d, eta_y0_d, eta_yL_d, g_x0_d, g_xL_d)
        if self.hparams['modeltype']=='DeepONet':
            u_hat = self.DeepONet_forward(theta_d, f_d, eta_y0_d, eta_yL_d, g_x0_d, g_xL_d)
        if self.hparams['modeltype']=='VarMiON':
            u_hat = self.VarMiON_forward(theta_d, f_d, eta_y0_d, eta_yL_d, g_x0_d, g_xL_d)
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO':
            u_hat = self.NGO_forward(F, d)
            F.detach().cpu()
            d.detach().cpu()
            self.phix.detach().cpu()
        if self.hparams['modeltype']=='FEM':
            u_hat = self.FEM_forward(F, d)
            F.detach().cpu()
            d.detach().cpu()
            self.phix.detach().cpu()
        if self.hparams['modeltype']=='projection':
            u_hat = self.projection_forward(u)
            # self.phix.detach().cpu()
        self.to('cpu')
        return u_hat
    
    def geometry(self):
        #Quadrature
        if self.hparams['quadrature']=='Gauss-Legendre':
            quad_OmegaT = GaussLegendreQuadrature(Q=self.hparams['Q'], n_elements=self.hparams['n_elements'])
            quad_Gamma_t = GaussLegendreQuadrature(Q=[self.hparams['Q'][1],self.hparams['Q'][2]], n_elements=[self.hparams['n_elements'][1],self.hparams['Q'][2]])
            quad_Gamma_x = GaussLegendreQuadrature(Q=[self.hparams['Q'][0],self.hparams['Q'][2]], n_elements=[self.hparams['n_elements'][0],self.hparams['Q'][2]])
            quad_Gamma_y = GaussLegendreQuadrature(Q=[self.hparams['Q'][0],self.hparams['Q'][1]], n_elements=[self.hparams['n_elements'][0],self.hparams['Q'][1]])
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

            # self.w_OmegaT = self.w_OmegaT, dtype=self.hparams['dtype'])
            # self.w_Gamma_t0 = self.w_Gamma_t0, dtype=self.hparams['dtype'])
            # self.w_Gamma_tT = self.w_Gamma_tT, dtype=self.hparams['dtype'])
            # self.w_Gamma_y0 = self.w_Gamma_y0, dtype=self.hparams['dtype'])
            # self.w_Gamma_yL = self.w_Gamma_yL, dtype=self.hparams['dtype'])
            # self.w_Gamma_x0 = self.w_Gamma_x0, dtype=self.hparams['dtype'])
            # self.w_Gamma_xL = self.w_Gamma_xL, dtype=self.hparams['dtype'])
            # self.w_t = self.w_t, dtype=self.hparams['dtype'])
            # self.w_x = self.w_x, dtype=self.hparams['dtype'])
            # self.w_y = self.w_y, dtype=self.hparams['dtype'])

            # self.xi_OmegaT = self.xi_OmegaT, dtype=self.hparams['dtype'])
            # self.xi_Gamma_t0 = self.xi_Gamma_t0, dtype=self.hparams['dtype'])
            # self.xi_Gamma_tT = self.xi_Gamma_tT, dtype=self.hparams['dtype'])
            # self.xi_Gamma_y0 = self.xi_Gamma_y0, dtype=self.hparams['dtype'])
            # self.xi_Gamma_yL = self.xi_Gamma_yL, dtype=self.hparams['dtype'])
            # self.xi_Gamma_x0 = self.xi_Gamma_x0, dtype=self.hparams['dtype'])
            # self.xi_Gamma_xL = self.xi_Gamma_xL, dtype=self.hparams['dtype'])
            # self.xi_t = self.xi_t, dtype=self.hparams['dtype'])
            # self.xi_x = self.xi_x, dtype=self.hparams['dtype'])
            # self.xi_y = self.xi_y, dtype=self.hparams['dtype'])

        #Outward normal
        self.n_y0 = np.array([0,-1])
        self.n_yL = np.array([0,1])
        self.n_x0 = np.array([-1,0])
        self.n_xL = np.array([1,0])

        #Loss quadrature
        if self.hparams['quadrature_L']=='Gauss-Legendre':
            quad_OmegaT_L = GaussLegendreQuadrature(Q=self.hparams['Q_L'], n_elements=self.hparams['n_elements_L'])
        self.w_OmegaT_L = torch.tensor(quad_OmegaT_L.w, dtype=self.hparams['dtype'], device=self.used_device)
        self.xi_OmegaT_L = quad_OmegaT_L.xi

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
                    T1 = -F@A_0 + self.Identity
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
                T1 = -F@A_0 + self.Identity
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
        self.phix = self.phix.to(self.used_device)
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