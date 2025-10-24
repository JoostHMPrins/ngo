import torch
import numpy as np
import pytorch_lightning as pl
import numpy as np
import opt_einsum
import os

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
        self.to(self.hparams['device'])
        self = self.to(self.hparams['dtype'])
        #compute_quadrature and quadrature
        self.compute_quadrature()
        self.rescale_quadrature_t()
        self.timestep = 0
        #Bases
        if self.hparams.get('POD',False)==False:
            self.basis_test = TensorizedBasis(self.hparams['test_bases']) 
            self.basis_trial = TensorizedBasis(self.hparams['trial_bases'])
        if self.hparams.get('POD',False)==True:
            self.basis_test = BSplineInterpolatedPOD2D(N_samples=self.hparams['N_samples_train'], d=self.hparams['d'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'], w=self.w_Omega, xi=self.xi_Omega, N=self.hparams['N'], device=self.device)
            self.basis_trial = BSplineInterpolatedPOD2D(N_samples=self.hparams['N_samples_train'], d=self.hparams['d'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'], w=self.w_Omega, xi=self.xi_Omega, N=self.hparams['N'], device=self.device)
        #Model type
        self.init_modeltype()
        #System net
        self.hparams['N_w_real'] = sum(p.numel() for p in self.parameters())
        self.systemnet = self.hparams['systemnet'](self.hparams).to(self.device)

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
            self.LBranch_f = LBranchNet(self.hparams, input_dim=len(self.xi_OmegaT), output_dim=self.hparams['N']).to(self.device)
            self.LBranch_eta = LBranchNet(self.hparams, input_dim=len(self.xi_Gamma_y0)+len(self.xi_Gamma_yL), output_dim=self.hparams['N']).to(self.device)
            self.LBranch_g = LBranchNet(self.hparams, input_dim=len(self.xi_Gamma_x0)+len(self.xi_Gamma_xL), output_dim=self.hparams['N']).to(self.device)
            self.LBranch_u0 = LBranchNet(self.hparams, input_dim=len(self.xi_Gamma_t0), output_dim=self.hparams['N']).to(self.device)
            self.forwardfunction = self.VarMiON_forward
            self.simforwardfunction = self.VarMiON_simforward
            self.hparams['input_shape'] = (1,)+(self.hparams['Q'])
            self.hparams['output_shape'] = (self.hparams['N'],self.hparams['N'])
            self.compute_Kronecker_factors()
            self.F_0 = torch.ones((1,)+self.hparams['Q'], dtype=self.dtype, device=self.device)
            x_0 = torch.randn(int(self.hparams['N']/2))
            x_0 = x_0/torch.sum(x_0**2)**(1/2)
            self.x_0 = torch.tensor(x_0, dtype=self.dtype, device=self.device)
            if os.path.exists(self.hparams['logdir']+'/x_0.pt'):
                print('loading x_0 from file')
                self.x_0 = torch.load(self.hparams['logdir']+'/x_0.pt').to(self.device)
            self.Identity = torch.eye(self.hparams['N'], dtype=self.dtype, device=self.device) 
        if self.hparams['modeltype']=='data NGO':
            self.forwardfunction = self.NGO_forward
            self.simforwardfunction = self.NGO_simforward
            self.hparams['input_shape'] = (2,)+self.hparams['h']
            self.hparams['output_shape'] = (self.hparams['N'],self.hparams['N'])
            self.compute_Kronecker_factors()
            self.compute_F_0_A_0()     
            self.Mnorm_0 = 1
            x_0 = torch.randn(int(self.hparams['N']/2))
            x_0 = x_0/torch.sum(x_0**2)**(1/2)
            self.x_0 = torch.tensor(x_0, dtype=self.dtype, device=self.device)
            if os.path.exists(self.hparams['logdir']+'/x_0.pt'):
                print('loading x_0 from file')
                self.x_0 = torch.load(self.hparams['logdir']+'/x_0.pt').to(self.device)
            self.Identity = torch.eye(self.hparams['N'], dtype=self.dtype, device=self.device) 
            self.C_0 = self.Identity[int(self.hparams['N']/2):,int(self.hparams['N']/2):]
        if self.hparams['modeltype']=='model NGO':
            self.forwardfunction = self.NGO_forward
            self.simforwardfunction = self.NGO_simforward
            self.hparams['input_shape'] = (1,self.hparams['N'],self.hparams['N'])
            self.hparams['output_shape'] = (self.hparams['N'],self.hparams['N'])
            self.compute_Kronecker_factors()
            self.compute_F_0_A_0() 
            self.Identity = torch.eye(self.hparams['N'], dtype=self.dtype, device=self.device)
            if self.hparams.get('Mnorm_scaling',False)==True:
                B_0 = self.A_0@self.M
                B_0 = B_0[int(self.A_0.shape[0]/2):,int(self.A_0.shape[1]/2):]
                Lambda, V = torch.linalg.eig(B_0)
                self.x_0 = V[Lambda==torch.amax(torch.abs(Lambda))]
        if self.hparams['modeltype']=='FEM':
            self.forwardfunction = self.FEM_forward
            self.simforwardfunction = self.FEM_simforward
            self.compute_Kronecker_factors()
            self.compute_F_0_A_0()
            self.Identity = torch.eye(self.hparams['N'], dtype=self.dtype, device=self.device)
        if self.hparams['modeltype']=='projection':
            self.forwardfunction = self.projection_forward
            self.simforwardfunction = self.projection_simforward
            self.compute_Kronecker_factors()
            self.compute_F_0_A_0()
            self.Identity = torch.eye(self.hparams['N'], dtype=self.dtype, device=self.device)

    def compute_Kronecker_factors(self):
        basis_test_x = self.hparams['test_bases'][1].forward(self.xi_x)
        basis_test_y = self.hparams['test_bases'][2].forward(self.xi_y)
        gradbasis_test_x = self.hparams['test_bases'][1].grad(self.xi_x)
        gradbasis_test_y = self.hparams['test_bases'][2].grad(self.xi_y)   
        basis_trial_x = self.hparams['trial_bases'][1].forward(self.xi_x)
        basis_trial_y = self.hparams['trial_bases'][2].forward(self.xi_y)
        gradbasis_trial_x = self.hparams['trial_bases'][1].grad(self.xi_x)
        gradbasis_trial_y = self.hparams['trial_bases'][2].grad(self.xi_y)
        basis_test_x0 = self.hparams['test_bases'][1].forward(0)[0]
        gradbasis_test_x0 = self.hparams['test_bases'][1].grad(0)[0]
        basis_trial_x0 = self.hparams['trial_bases'][1].forward(0)[0]
        gradbasis_trial_x0 = self.hparams['trial_bases'][1].grad(0)[0]
        basis_test_xL = self.hparams['test_bases'][1].forward(1)[0]
        gradbasis_test_xL = self.hparams['test_bases'][1].grad(1)[0]
        basis_trial_xL = self.hparams['trial_bases'][1].forward(1)[0]
        gradbasis_trial_xL = self.hparams['trial_bases'][1].grad(1)[0]
        basis_test_y0 = self.hparams['test_bases'][2].forward(0)[0]
        basis_test_yL = self.hparams['test_bases'][2].forward(1)[0]
        basis_test_t = self.hparams['test_bases'][0].forward(self.xi_t)
        gradbasis_test_t = 1/self.hparams['Dt']*self.hparams['test_bases'][0].grad(self.xi_t)
        basis_trial_t = self.hparams['trial_bases'][0].forward(self.xi_t)
        basis_test_tT = self.hparams['test_bases'][0].forward(1)[0]
        basis_trial_tT = self.hparams['trial_bases'][0].forward(1)[0]
        basis_test_t0 = self.hparams['test_bases'][0].forward(0)[0]
        basis_trial_t0 = self.hparams['trial_bases'][0].forward(0)[0]
        basis_test_tT = self.hparams['test_bases'][0].forward(1)[0]
        basis_trial_tT = self.hparams['trial_bases'][0].forward(1)[0]
        self.K_x_psipsiphi = opt_einsum.contract('q,qb,qe,qh->beh', self.w_x, basis_test_x, basis_test_x, basis_trial_x)
        self.K_y_psipsiphi = opt_einsum.contract('q,qc,qf,qi->cfi', self.w_y, basis_test_y, basis_test_y, basis_trial_y)
        self.K_x_psidpsidphi = opt_einsum.contract('q,qb,qe,qh->beh', self.w_x, basis_test_x, gradbasis_test_x, gradbasis_trial_x)
        self.K_y_psidpsidphi = opt_einsum.contract('q,qc,qf,qi->cfi', self.w_y, basis_test_y, gradbasis_test_y, gradbasis_trial_y)
        self.K_x_psipsiphi_x0 = opt_einsum.contract('b,e,h->beh', basis_test_x0, basis_test_x0, basis_trial_x0)
        self.K_x_psipsiphi_xL = opt_einsum.contract('b,e,h->beh', basis_test_xL, basis_test_xL, basis_trial_xL)
        self.K_x_psipsidphi_x0 = opt_einsum.contract('b,e,h->beh', basis_test_x0, basis_test_x0, gradbasis_trial_x0)
        self.K_x_psidpsiphi_x0 = opt_einsum.contract('b,e,h->beh', basis_test_x0, gradbasis_test_x0, basis_trial_x0)
        self.K_x_psipsidphi_xL = opt_einsum.contract('b,e,h->beh', basis_test_xL, basis_test_xL, gradbasis_trial_xL)
        self.K_x_psidpsiphi_xL = opt_einsum.contract('b,e,h->beh', basis_test_xL, gradbasis_test_xL, basis_trial_xL)    
        self.K_x_psiphi = opt_einsum.contract('q,qe,qh->eh', self.w_x, basis_test_x, basis_trial_x)
        self.K_x_psiphi_inv = np.linalg.inv(self.K_x_psiphi)
        self.K_y_psiphi = opt_einsum.contract('q,qf,qi->fi', self.w_y, basis_test_y, basis_trial_y)
        self.K_y_psiphi_inv = np.linalg.inv(self.K_y_psiphi)
        self.M_x_psipsi = opt_einsum.contract('q,qb,qe->be', self.w_x, basis_test_x, basis_test_x)
        self.M_y_psipsi = opt_einsum.contract('q,qc,qf->cf', self.w_y, basis_test_y, basis_test_y)
        self.M_x_phiphi = opt_einsum.contract('q,qb,qh->bh', self.w_x, basis_trial_x, basis_trial_x)
        self.M_y_phiphi = opt_einsum.contract('q,qc,qi->ci', self.w_y, basis_trial_y, basis_trial_y)
        self.M_x_psiphi = opt_einsum.contract('q,qb,qh->bh', self.w_x, basis_test_x, basis_trial_x)
        self.M_y_psiphi = opt_einsum.contract('q,qc,qi->ci', self.w_y, basis_test_y, basis_trial_y)
        self.M_x_psipsi_inv = np.linalg.pinv(self.M_x_psipsi)
        self.M_y_psipsi_inv = np.linalg.pinv(self.M_y_psipsi)
        self.M_x_phiphi_inv = np.linalg.pinv(self.M_x_phiphi)
        self.M_y_phiphi_inv = np.linalg.pinv(self.M_y_phiphi)
        self.M_y_psipsi_y0 = opt_einsum.contract('c,f->cf', basis_test_y0, basis_test_y0)
        self.M_y_psipsi_yL = opt_einsum.contract('c,f->cf', basis_test_yL, basis_test_yL)
        self.M_x_psidpsi_x0 = opt_einsum.contract('c,f->cf', basis_test_x0, gradbasis_test_x0)
        self.M_x_psidpsi_xL = opt_einsum.contract('c,f->cf', basis_test_xL, gradbasis_test_xL)
        self.K_t_psipsiphi = opt_einsum.contract('q,qa,qd,qg->adg', self.w_t, basis_test_t, basis_test_t, basis_trial_t)
        self.K_t_dpsiphi = opt_einsum.contract('q,qd,qg->dg', self.w_t, gradbasis_test_t, basis_trial_t)
        self.K_t_psiphi_tT = opt_einsum.contract('d,g->dg', basis_test_tT, basis_trial_tT)
        self.M_t_psipsi = opt_einsum.contract('q,qa,qg->ag', self.w_t, basis_test_t, basis_test_t)
        self.M_t_phiphi = opt_einsum.contract('q,qa,qg->ag', self.w_t, basis_trial_t, basis_trial_t)
        self.M_t_psipsi_inv = np.linalg.pinv(self.M_t_psipsi)
        self.M_t_phiphi_inv = np.linalg.pinv(self.M_t_phiphi)
        self.M_phiphi = np.kron(np.kron(self.M_t_phiphi, self.M_x_phiphi), self.M_y_phiphi)/self.hparams['Dt']
        self.M_t_psipsi_t0 = opt_einsum.contract('a,d->ad', basis_test_t0, basis_test_t0)
        self.M_t_psipsi_tT = opt_einsum.contract('a,d->ad', basis_test_tT, basis_test_tT)
        self.M_t_psiphi_t0 = opt_einsum.contract('a,d->ad', basis_test_t0, basis_trial_t0)
        self.M_t_psiphi_tT = opt_einsum.contract('a,d->ad', basis_test_tT, basis_trial_tT)
        self.M_t_phiphi_tT = opt_einsum.contract('a,d->ad', basis_trial_tT, basis_trial_tT)
        self.M_t_psiphi_t0T = opt_einsum.contract('a,d->ad', basis_test_t0, basis_trial_tT)
        self.M_lr = np.kron(np.kron(self.M_t_psiphi_t0T, self.K_x_psiphi), self.K_y_psiphi)/self.hparams['Dt']
        #self.M_lr = torch.tensor(self.M_lr, dtype=self.dtype, device=self.device)
        self.M_rr = np.kron(np.kron(self.M_t_psiphi_tT, self.K_x_psiphi), self.K_y_psiphi)/self.hparams['Dt']
        self.M_rr22 = self.M_rr[int(self.hparams['N']/2):,int(self.hparams['N']/2):]
        self.M_rr = torch.tensor(self.M_rr, dtype=self.dtype, device=self.device)
        Lambda_22, T_22 = np.linalg.eig(self.M_rr22)
        self.M_rr22_05 = T_22 @ np.diag(Lambda_22**0.5) @ T_22.T
        self.M_rr22_05 = torch.tensor(self.M_rr22_05, dtype=self.dtype, device=self.device)
        self.M_rr22_05_inv = T_22 @ np.diag(1/Lambda_22**0.5) @ T_22.T
        self.M_rr22_05_inv = torch.tensor(self.M_rr22_05_inv, dtype=self.dtype, device=self.device)

        K_c = self.K_t_psiphi_tT - self.K_t_dpsiphi
        K_c = np.kron(K_c, self.K_x_psiphi)
        self.K_c = np.kron(K_c, self.K_y_psiphi)
        K_lmn = self.K_x_psidpsidphi - self.K_x_psipsidphi_xL + self.K_x_psipsidphi_x0 - self.K_x_psidpsiphi_xL + self.K_x_psidpsiphi_x0
        K_lmn = np.kron(K_lmn, self.K_y_psipsiphi)
        K_lmn = K_lmn + np.kron(self.K_x_psipsiphi, self.K_y_psidpsidphi)
        self.K_lmn = np.kron(self.K_t_psipsiphi, K_lmn)
        if self.hparams.get('gamma_stabilization',0)!=0:
            self.K_lmn += self.hparams['gamma_stabilization']*np.kron(np.kron(self.K_t_psipsiphi, self.K_x_psipsiphi_x0 + self.K_x_psipsiphi_xL), self.K_y_psipsiphi)


    def discretize_input_functions(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        theta_q = discretize_functions(theta, self.xi_OmegaT_scaled)
        theta_x0_q = discretize_functions(theta, self.xi_Gamma_x0_scaled)
        theta_xL_q = discretize_functions(theta, self.xi_Gamma_xL_scaled)
        f_q = discretize_functions(f, self.xi_OmegaT_scaled)
        u0_t0_q = discretize_functions(u0, self.xi_Gamma_t0_scaled)
        # if self.hparams.get('project_initialcondition',False)==True:
        #     u0_m = self.project_on_psi_t0(u0_t0_q)
        #     u0_t0_q = opt_einsum.contract('Nn,qn->Nq', u0_m, self.basis_trial.forward(self.xi_Gamma_t0))
        eta_y0_q = discretize_functions(eta_y0, self.xi_Gamma_y0_scaled)
        eta_yL_q = discretize_functions(eta_yL, self.xi_Gamma_yL_scaled)
        g_x0_q = discretize_functions(g_x0, self.xi_Gamma_x0_scaled)
        g_xL_q = discretize_functions(g_xL, self.xi_Gamma_xL_scaled)
        # theta_q = theta(self.xi_OmegaT_scaled)
        # theta_x0_q = theta(self.xi_Gamma_x0_scaled)
        # theta_xL_q = theta(self.xi_Gamma_xL_scaled)
        # f_q = f(self.xi_OmegaT_scaled)
        # u0_t0_q = u0(self.xi_Gamma_t0_scaled)
        # eta_y0_q = eta_y0(self.n_x0, self.xi_Gamma_y0_scaled)
        # eta_yL_q = eta_yL(self.n_xL, self.xi_Gamma_yL_scaled)
        # g_x0_q = g_x0(self.xi_Gamma_x0_scaled)
        # g_xL_q = g_xL(self.xi_Gamma_xL_scaled) 
        return theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q
    
    def discretize_output_function(self, u):
        # u_q = discretize_functions(u, self.xi_OmegaT_L_scaled)
        # u_q = discretize_functions(u, self.xi_Gamma_tT_scaled)
        u_q = u(self.xi_OmegaT_L_scaled)
        return u_q
    
    def project_input_function(self, u_q):
        basis_test_OmegaT = self.basis_test.forward(self.xi_OmegaT)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT, basis_test_OmegaT, u_q)
        M_inv = np.kron(np.kron(self.M_t_psipsi_inv, self.M_x_psipsi_inv), self.M_y_psipsi_inv)
        u_m = opt_einsum.contract('mn,Nm->Nn', M_inv, d)
        return u_m

    def project_on_psi_t0(self, u_q):
        basis_test_t0 = self.basis_test.forward(self.xi_Gamma_t0)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_t0, basis_test_t0, u_q)
        M = np.kron(np.kron(self.M_t_psipsi_t0, self.M_x_psipsi), self.M_y_psipsi)
        M_inv = np.linalg.pinv(M)
        u_m = opt_einsum.contract('mn,Nm->Nn', M_inv, d)
        return u_m
    
    def project_on_psi_tT(self, u_q):
        basis_test_tT = self.basis_test.forward(self.xi_Gamma_tT)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_tT, basis_test_tT, u_q)
        # M = np.kron(np.kron(self.M_t_psipsi_tT, self.M_x_psipsi), self.M_y_psipsi)
        M = np.kron(np.kron(self.M_t_psipsi_tT, self.M_x_psipsi), self.M_y_psipsi)
        M_inv = np.linalg.pinv(M)
        u_m = opt_einsum.contract('mn,Nm->Nn', M_inv, d)
        return u_m
    
    def project_output_function(self, u_q):
        basis_trial_OmegaT_L = self.basis_trial.forward(self.xi_OmegaT_L)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT_L, basis_trial_OmegaT_L, u_q)
        M_inv = np.kron(np.kron(self.M_t_phiphi_inv, self.M_x_phiphi_inv), self.M_y_phiphi_inv)
        u_m = opt_einsum.contract('mn,Nm->Nn', M_inv, d)
        return u_m

    def compute_F_direct(self, theta_q, theta_x0_q, theta_xL_q):
        if  self.hparams['modeltype']=='data NGO':
            basis_test_OmegaT = self.basis_test.forward(self.xi_OmegaT)
            basis_test_x0 = self.basis_test.forward(self.xi_Gamma_x0)
            basis_test_xL = self.basis_test.forward(self.xi_Gamma_xL)
            basis_test_tT = self.basis_test.forward(self.xi_Gamma_tT)
            # F = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT, basis_test_OmegaT, theta_q)
            # F = F.reshape((theta_q.shape[0],)+self.hparams['h'])
            F = np.zeros((theta_q.shape[0],2)+self.hparams['h'])
            F1 = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT, basis_test_OmegaT, theta_q).reshape((theta_q.shape[0],)+self.hparams['h'])
            F[:,0,:] = F1
            F2 = opt_einsum.contract('q,qm->m', self.w_Gamma_tT, basis_test_tT).reshape((1,)+self.hparams['h'])
            F[:,1,:] = F2
            # F += F1
        if  self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM' or self.hparams['modeltype']=='projection':
            basis_trial = self.basis_trial.forward(self.xi_OmegaT)
            gradbasis_test = self.basis_test.grad(self.xi_OmegaT)[:,:,1:]
            gradbasis_trial = self.basis_trial.grad(self.xi_OmegaT)[:,:,1:]
            ddtbasis_test = 1/self.hparams['Dt']*self.basis_test.grad(self.xi_OmegaT)[:,:,0]
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
            F = opt_einsum.contract('q,Nq,qmx,qnx->Nmn', self.w_OmegaT, theta_q, gradbasis_test, gradbasis_trial)
            F += -opt_einsum.contract('q,qn,qm->mn', self.w_OmegaT, basis_trial, ddtbasis_test)
            F += -opt_einsum.contract('q,qm,x,Nq,qnx->Nmn', self.w_Gamma_x0, basis_test_x0, self.n_x0, theta_x0_q, gradbasis_trial_x0)
            F += -opt_einsum.contract('q,qm,x,Nq,qnx->Nmn', self.w_Gamma_xL, basis_test_xL, self.n_xL, theta_xL_q, gradbasis_trial_xL)
            F += -opt_einsum.contract('q,qn,x,Nq,qmx->Nmn', self.w_Gamma_x0, basis_trial_x0, self.n_x0, theta_x0_q, gradbasis_test_x0)
            F += -opt_einsum.contract('q,qn,x,Nq,qmx->Nmn', self.w_Gamma_xL, basis_trial_xL, self.n_xL, theta_xL_q, gradbasis_test_xL)
            F += opt_einsum.contract('q,qm,qn->mn', self.w_Gamma_tT, basis_test_tT, basis_trial_tT)
            if self.hparams.get('gamma_stabilization',0)!=0:
                F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_x0, basis_test_x0, theta_x0_q, basis_trial_x0)
                F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_xL, basis_test_xL, theta_xL_q, basis_trial_xL)
        return F

    def compute_F_tensorized(self, theta_q):
        theta_l = self.project_input_function(theta_q)
        # K_c = self.K_t_psiphi_tT - self.K_t_dpsiphi
        # K_c = np.kron(K_c, self.K_x_psiphi)
        # K_c = np.kron(K_c, self.K_y_psiphi)
        # K_lmn = self.K_x_psidpsidphi - self.K_x_psipsidphi_xL + self.K_x_psipsidphi_x0 - self.K_x_psidpsiphi_xL + self.K_x_psidpsiphi_x0
        # K_lmn = np.kron(K_lmn, self.K_y_psipsiphi)
        # K_lmn = K_lmn + np.kron(self.K_x_psipsiphi, self.K_y_psidpsidphi)
        # K_lmn = np.kron(self.K_t_psipsiphi, K_lmn)
        # if self.hparams.get('gamma_stabilization',0)!=0:
        #     K_lmn += self.hparams['gamma_stabilization']*np.kron(np.kron(self.K_t_psipsiphi, self.K_x_psipsiphi_x0 + self.K_x_psipsiphi_xL), self.K_y_psipsiphi)
        K = opt_einsum.contract('Nl,lmn->Nmn', theta_l, self.K_lmn) + self.K_c
        return K
    
    def compute_stability_criterion(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0):
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
        F = torch.tensor(self.compute_F(theta_q, theta_x0_q, theta_xL_q), dtype=self.dtype, device=self.device)/self.hparams['Dt']
        d = torch.tensor(self.compute_d(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q), dtype=self.dtype, device=self.device)/self.hparams['Dt']
        d_x = torch.tensor(self.compute_d_x(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q), dtype=self.dtype, device=self.device)/self.hparams['Dt']
        C = torch.tensor(self.compute_C(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q), dtype=self.dtype, device=self.device)/self.hparams['Dt']
        C_m = torch.tensor(self.compute_C_m(theta_x0_q, theta_xL_q), dtype=self.dtype, device=self.device)/self.hparams['Dt']
        theta_bar = torch.tensor(opt_einsum.contract('q,Nq->N', self.w_OmegaT, theta_q), dtype=self.dtype, device=self.device)/self.hparams['Dt']
        u_0_m = torch.tensor(self.project_on_psi_tT(u0_t0_q), dtype=self.dtype, device=self.device)
        if self.hparams['modeltype']=='FEM':
            A_hat = torch.linalg.inv(F)
        if self.hparams['modeltype']=='data NGO' or self.hparams['modeltype']=='model NGO':           
            if self.hparams.get('scaling_equivariance',False)==True:
                F = F/theta_bar[:,None,None,None,None]
            if self.hparams.get('Neumannseries', False)==False:
                A_hat = self.systemnet.forward(F)*self.hparams['Dt']
            if self.hparams.get('Neumannseries', False)==True:
                T = torch.zeros(F.shape, dtype=self.dtype, device=self.device)
                Ti = self.Identity
                T1 = -F@self.A_0 + self.Identity
                for i in range(0, self.hparams['Neumannseries_order']):
                    Ti = T1@Ti
                    T = T + Ti
                A_hat = self.A_0@(self.Identity + T + self.systemnet.forward(T1))
            # if self.hparams.get('PD', False)==True:
            #     L = torch.tril(A_hat[:,0])
            #     diag_indices = torch.arange(self.hparams['N'], dtype=torch.long, device=self.device)
            #     L_diag = L[:,diag_indices,diag_indices]
            #     # L_diag_pos = torch.nn.functional.softplus(L_diag) + 1e-6
            #     L_diag_pos = torch.abs(L_diag) + 1e-6
            #     L = L.clone()  # avoid in-place on input
            #     L[:,diag_indices,diag_indices] = L_diag_pos
            #     S = L @ L.transpose(-1, -2)
            #     N = A_hat[:,1]
            #     N = 0.5*(N - N.transpose(-1, -2))
            #     A_hat = S + N
            if self.hparams.get('scaling_equivariance',False)==True:
                A_hat = A_hat/theta_bar[:,None,None]
            if self.hparams.get('Mnorm_scaling',False)==True:
                # B = A_hat@self.M_lr
                B = A_hat@torch.tensor(self.M_lr, dtype=self.dtype, device=self.device)
                B = B[:,int(A_hat.shape[1]/2):,int(A_hat.shape[2]/2):]
                M_rr22_05TBM_rr22_05_inv = self.M_rr22_05 @ B @ self.M_rr22_05_inv
                Mnorm, v = two_norm_power_iteration(B=M_rr22_05TBM_rr22_05_inv, norm2_0=self.Mnorm_0, x_0=self.x_0, tol=self.hparams['tol'], max_iterations=self.hparams['max_iterations'], dtype=self.dtype, device=self.device, model=self)
                scaling = torch.maximum(self.hparams['Mnorm_max']*torch.ones_like(Mnorm),Mnorm)/self.hparams['Mnorm_max']
                A_hat = A_hat/scaling[:,None,None]
            u_m_hat = opt_einsum.contract('nij,nj->ni', A_hat, d)
            if self.hparams['massconservation']==True:# and self.hparams['dissipation']==False:
                C_hat = opt_einsum.contract('Nn,Nn->N', u_m_hat, C_m)
                l = (C - C_hat)/opt_einsum.contract('Nn,Nn->N', C_m, C_m)
                u_m_hat = u_m_hat + l[:,None]*C_m
            if self.hparams['dissipation']==True and self.hparams['massconservation']==False:
                u_m_hat = conserve_energy(self, F, d_x, u_0_m, u_m_hat)
            if self.hparams['massconservation']==True and self.hparams['dissipation']==True:
                u_m_hat = conserve_mass_and_energy(self, d_x, u_0_m, C_m, C, u_m_hat)

        A_hat = A_hat.detach().cpu().numpy()
        A_hat = np.array(A_hat, dtype=np.float64)
        # M_lr = self.M_lr.detach().cpu().numpy()
        # M_lr = np.array(M_lr, dtype=np.float64)
        self.M_lr = np.kron(np.kron(self.M_t_psiphi_t0T, self.K_x_psiphi), self.K_y_psiphi)/self.hparams['Dt']
        B = A_hat@self.M_lr
        # B = B[0]
        # M_rr = self.M_rr
        B = B[0,100:,100:]
        self.M_rr = np.kron(np.kron(self.M_t_psiphi_tT, self.K_x_psiphi), self.K_y_psiphi)/self.hparams['Dt']
        M_rr = self.M_rr[100:,100:]

        # print(np.amax(np.linalg.eigvals((np.linalg.inv(M_rr)))))

        Lambda, T = np.linalg.eig(M_rr)
        M_rr_05 = T @ np.diag(Lambda**0.5) @ T.T
        # M_rr_05_inv = T @ np.diag(1/np.amax(Lambda**0.5)) @ T.T
        M_rr_05_inv = np.linalg.inv(M_rr_05)
        # print(np.linalg.norm(M_rr_05@B@np.linalg.inv(M_rr_05), ord='fro'))

        # spectrum = np.sort(np.linalg.eigvals(B))
        spectrum = np.sort(np.sqrt(np.linalg.eigvals(np.linalg.inv(M_rr) @ B.T @ M_rr @ B)))
        # spectrum = np.sort(np.sqrt(np.linalg.eigvals(M_rr_05_inv.T @ B.T @ M_rr @ B @ M_rr_05_inv)))

        return spectrum
    
    def compute_D(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        u_m_hat = self.simforward_coefficients(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u)
        theta_q = theta(self.xi_OmegaT_scaled)
        # theta_x0_q = discretize_functions(theta, self.xi_Gamma_x0_scaled)
        # theta_xL_q = discretize_functions(theta, self.xi_Gamma_xL_scaled)
        # basis_trial = self.basis_trial.forward(self.xi_OmegaT)
        # gradbasis_test = self.basis_test.grad(self.xi_OmegaT)[:,:,1:]
        # gradbasis_trial = self.basis_trial.grad(self.xi_OmegaT)[:,:,1:]
        # basis_test_x0 = self.basis_test.forward(self.xi_Gamma_x0)
        # basis_test_xL = self.basis_test.forward(self.xi_Gamma_xL)
        # gradbasis_test_x0 = self.basis_test.grad(self.xi_Gamma_x0)[:,:,1:]
        # gradbasis_test_xL = self.basis_test.grad(self.xi_Gamma_xL)[:,:,1:]
        # basis_trial_x0 = self.basis_trial.forward(self.xi_Gamma_x0)
        # basis_trial_xL = self.basis_trial.forward(self.xi_Gamma_xL)
        # gradbasis_trial_x0 = self.basis_trial.grad(self.xi_Gamma_x0)[:,:,1:]
        # gradbasis_trial_xL = self.basis_trial.grad(self.xi_Gamma_xL)[:,:,1:]
        # F_x = opt_einsum.contract('q,Nq,qmx,qnx->Nmn', self.w_OmegaT, theta_q, gradbasis_test, gradbasis_trial)
        # F_x += -opt_einsum.contract('q,qm,x,Nq,qnx->Nmn', self.w_Gamma_x0, basis_test_x0, self.n_x0, theta_x0_q, gradbasis_trial_x0)
        # F_x += -opt_einsum.contract('q,qm,x,Nq,qnx->Nmn', self.w_Gamma_xL, basis_test_xL, self.n_xL, theta_xL_q, gradbasis_trial_xL)
        # F_x += -opt_einsum.contract('q,qn,x,Nq,qmx->Nmn', self.w_Gamma_x0, basis_trial_x0, self.n_x0, theta_x0_q, gradbasis_test_x0)
        # F_x += -opt_einsum.contract('q,qn,x,Nq,qmx->Nmn', self.w_Gamma_xL, basis_trial_xL, self.n_xL, theta_xL_q, gradbasis_test_xL)
        # if self.hparams.get('gamma_stabilization',0)!=0:
        #     F_x += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_x0, basis_test_x0, theta_x0_q, basis_trial_x0)
        #     F_x += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_xL, basis_test_xL, theta_xL_q, basis_trial_xL)
        theta_l = self.project_input_function(theta_q)
        F_x = opt_einsum.contract('Nl,lmn->Nmn', theta_l, self.K_lmn)# + self.K_c
        uTFu = opt_einsum.contract('Nm,Nmn,Nn->N', u_m_hat, F_x, u_m_hat)
        D = uTFu
        return D
    
    def compute_F(self, theta_q, theta_x0_q, theta_xL_q):
        if self.hparams['project_materialparameters']==True:
            F = self.compute_F_tensorized(theta_q)
        if self.hparams['project_materialparameters']==False:
            F = self.compute_F_direct(theta_q, theta_x0_q, theta_xL_q)
        return F

    def compute_F_0_A_0(self):
        theta_q = np.ones((1,self.xi_OmegaT.shape[0]))
        theta_x0_q = np.ones((1,self.xi_Gamma_x0.shape[0]))
        theta_xL_q = np.ones((1,self.xi_Gamma_xL.shape[0]))
        F_0 = self.compute_F(theta_q, theta_x0_q, theta_xL_q)/self.hparams['Dt']
        if self.hparams['modeltype']=='data NGO':
            self.F_0 = torch.tensor(F_0, dtype=self.dtype, device=self.device)
        if self.hparams['modeltype']=='model NGO':
            self.F_0 = torch.tensor(F_0[0], dtype=self.dtype, device=self.device)
            A_0 = np.linalg.inv(F_0[0])
            self.A_0 = torch.tensor(A_0, dtype=self.dtype, device=self.device)

    def compute_d(self, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        basis_test_OmegaT = self.basis_test.forward(self.xi_OmegaT)
        basis_test_Gamma_y0 = self.basis_test.forward(self.xi_Gamma_y0)
        basis_test_Gamma_yL = self.basis_test.forward(self.xi_Gamma_yL)
        basis_test_Gamma_x0 = self.basis_test.forward(self.xi_Gamma_x0)
        basis_test_Gamma_xL = self.basis_test.forward(self.xi_Gamma_xL)
        gradbasis_test_Gamma_x0 = self.basis_test.grad(self.xi_Gamma_x0)[:,:,1:]
        gradbasis_test_Gamma_xL = self.basis_test.grad(self.xi_Gamma_xL)[:,:,1:]
        basis_test_Gamma_t0 = self.basis_test.forward(self.xi_Gamma_t0)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT, basis_test_OmegaT, f_q)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_y0, basis_test_Gamma_y0, eta_y0_q)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_yL, basis_test_Gamma_yL, eta_yL_q)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_x0, self.n_x0, gradbasis_test_Gamma_x0, g_x0_q)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_xL, self.n_xL, gradbasis_test_Gamma_xL, g_xL_q)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_t0, basis_test_Gamma_t0, u0_t0_q)
        if self.hparams.get('gamma_stabilization',0)!=0:
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_x0, basis_test_Gamma_x0, g_x0_q)
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_xL, basis_test_Gamma_xL, g_xL_q)
        return d

    def compute_d_x(self, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        basis_test_OmegaT = self.basis_test.forward(self.xi_OmegaT)
        basis_test_Gamma_y0 = self.basis_test.forward(self.xi_Gamma_y0)
        basis_test_Gamma_yL = self.basis_test.forward(self.xi_Gamma_yL)
        basis_test_Gamma_x0 = self.basis_test.forward(self.xi_Gamma_x0)
        basis_test_Gamma_xL = self.basis_test.forward(self.xi_Gamma_xL)
        gradbasis_test_Gamma_x0 = self.basis_test.grad(self.xi_Gamma_x0)[:,:,1:]
        gradbasis_test_Gamma_xL = self.basis_test.grad(self.xi_Gamma_xL)[:,:,1:]
        basis_test_Gamma_t0 = self.basis_test.forward(self.xi_Gamma_t0)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_OmegaT, basis_test_OmegaT, f_q)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_y0, basis_test_Gamma_y0, eta_y0_q)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_yL, basis_test_Gamma_yL, eta_yL_q)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_x0, self.n_x0, gradbasis_test_Gamma_x0, g_x0_q)
        d -= opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_xL, self.n_xL, gradbasis_test_Gamma_xL, g_xL_q)
        # d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_t0, basis_test_Gamma_t0, u0_t0_q)
        if self.hparams.get('gamma_stabilization',0)!=0:
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_x0, basis_test_Gamma_x0, g_x0_q)
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_xL, basis_test_Gamma_xL, g_xL_q)
        return d

    #Conserved quantity
    def compute_C(self, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        C = opt_einsum.contract('q,Nq->N', self.w_OmegaT, f_q)
        C += opt_einsum.contract('q,Nq->N', self.w_Gamma_y0, eta_y0_q)
        C += opt_einsum.contract('q,Nq->N', self.w_Gamma_yL, eta_yL_q)
        C += opt_einsum.contract('q,Nq->N', self.w_Gamma_t0, u0_t0_q)
        if self.hparams.get('stabilization_in_massconservation',False)==True:
            C += self.hparams['gamma_stabilization']*opt_einsum.contract('q,Nq->N', self.w_Gamma_x0, g_x0_q)
            C += self.hparams['gamma_stabilization']*opt_einsum.contract('q,Nq->N', self.w_Gamma_xL, g_xL_q)    
        return C
   
    #Conserved quantity per DOF
    def compute_C_m(self, theta_x0_q, theta_xL_q):
        gradbasis_trial_Gamma_x0 = self.basis_trial.grad(self.xi_Gamma_x0)[:,:,1:]
        gradbasis_trial_Gamma_xL = self.basis_trial.grad(self.xi_Gamma_xL)[:,:,1:]
        basis_trial_x0 = self.basis_trial.forward(self.xi_Gamma_x0)
        basis_trial_xL = self.basis_trial.forward(self.xi_Gamma_xL)
        C_m = -opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_x0, self.n_x0, gradbasis_trial_Gamma_x0, theta_x0_q)
        C_m += -opt_einsum.contract('q,x,qmx,Nq->Nm', self.w_Gamma_xL, self.n_xL, gradbasis_trial_Gamma_xL, theta_xL_q)
        C_m += opt_einsum.contract('q,qm->m', self.w_Gamma_tT, self.basis_trial.forward(self.xi_Gamma_tT))
        if self.hparams.get('stabilization_in_massconservation',False)==True:
            C_m += self.hparams['gamma_stabilization']*opt_einsum.contract('q,Nq,qm->Nm', self.w_Gamma_x0, theta_x0_q, basis_trial_x0)
            C_m += self.hparams['gamma_stabilization']*opt_einsum.contract('q,Nq,qm->Nm', self.w_Gamma_xL, theta_xL_q, basis_trial_xL)
        return C_m

    def compute_C_s(self):
        basis_test_x0 = self.basis_test.forward(self.xi_Gamma_x0)
        basis_test_xL = self.basis_test.forward(self.xi_Gamma_xL)
        gradbasis_test_Gamma_x0 = self.basis_test.grad(self.xi_Gamma_x0)[:,:,1:]
        gradbasis_test_Gamma_xL = self.basis_test.grad(self.xi_Gamma_xL)[:,:,1:]
        ndotgradphi = opt_einsum.contract('q,x,qmx->m', self.w_Gamma_x0, self.n_x0, gradbasis_test_Gamma_x0)
        ndotgradphi += opt_einsum.contract('q,x,qmx->m', self.w_Gamma_xL, self.n_xL, gradbasis_test_Gamma_xL)
        phi = opt_einsum.contract('q,qm->m', self.w_Gamma_x0, basis_test_x0)
        phi += opt_einsum.contract('q,qm->m', self.w_Gamma_xL, basis_test_xL)
        C_s = 2*np.sqrt(np.sum(ndotgradphi**2)/np.sum(phi**2))
        print("C_s")
        print(C_s)
        # self.hparams['gamma_stabilization'] = C_s
        return C_s
        
    def NN_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        theta_q = theta_q.reshape((theta_q.shape[0],)+(self.hparams['Q']))
        f_q = f_q.reshape((f_q.shape[0],)+(self.hparams['Q']))
        eta_y0_q = eta_y0_q.reshape((eta_y0_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][1],))
        eta_yL_q = eta_yL_q.reshape((eta_yL_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][1],))
        g_x0_q = g_x0_q.reshape((g_x0_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][2],))
        g_xL_q = g_xL_q.reshape((g_xL_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][2],))  
        u0_t0_q = u0_t0_q.reshape((u0_t0_q.shape[0],)+(self.hparams['Q'][1],)+(self.hparams['Q'][2],))       
        eta_q = torch.zeros((eta_y0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        eta_q[:,:,:,0] = eta_y0_q
        eta_q[:,:,:,-1] = eta_yL_q
        g_q = torch.zeros((g_x0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        g_q[:,:,0,:] = g_x0_q
        g_q[:,:,-1,:] = g_xL_q
        u0_t0 = torch.zeros((u0_t0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        u0_t0[:,0,:,:] = u0_t0_q
        u0_t0_q = u0_t0
        inputfuncs = torch.stack((theta_q,f_q,eta_q,g_q,u0_t0_q), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta_q.flatten(-3,-1),f_q.flatten(-3,-1),eta_y0_q,eta_yL_q,g_x0_q,g_xL_q,u0_t0_q), dim=1)
        u_q_hat = self.systemnet.forward(inputfuncs).reshape((theta_q.shape[0],)+(np.prod(self.hparams['Q_L']),))
        return u_q_hat
    
    def NN_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
        theta_q = torch.tensor(theta_q, dtype=self.dtype, device=self.device)
        f_q = torch.tensor(f_q, dtype=self.dtype, device=self.device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.dtype, device=self.device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.dtype, device=self.device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.dtype, device=self.device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.dtype, device=self.device)   
        u0_t0_q = torch.tensor(u0_t0_q, dtype=self.dtype, device=self.device)
        if self.timestep>0:
            u0_t0_q = torch.tensor(self.u_q_hat, dtype=self.dtype, device=self.device)
            u0_t0_q = u0_t0_q.reshape(((u0_t0_q.shape[0],)+self.hparams['Q']))
            u0_t0_q = u0_t0_q[:,-1]
        if self.hparams['systemnet']==FNO:
            bs = 100
            if theta_q.shape[0]>bs:
                u_q_hat = np.zeros((theta_q.shape[0],)+(np.prod(self.hparams['Q_L']),))
                for i in range(int(theta_q.shape[0]/bs)):
                    u_q_hat[bs*i:bs*(i+1)] = self.NN_forward(theta_q[bs*i:bs*(i+1)], f_q[bs*i:bs*(i+1)], eta_y0_q[bs*i:bs*(i+1)], eta_yL_q[bs*i:bs*(i+1)], g_x0_q[bs*i:bs*(i+1)], g_xL_q[bs*i:bs*(i+1)], u0_t0_q[bs*i:bs*(i+1)]).detach().cpu().numpy()
            else:
                u_q_hat = self.NN_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q).detach().cpu().numpy()
        else:
            u_q_hat = self.NN_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q).detach().cpu().numpy()
        self.u_q_hat = u_q_hat
        return u_q_hat
    
    def DeepONet_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        theta_q = theta_q.reshape((theta_q.shape[0],)+(self.hparams['Q']))
        f_q = f_q.reshape((f_q.shape[0],)+(self.hparams['Q']))
        eta_y0_q = eta_y0_q.reshape((eta_y0_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][1],))
        eta_yL_q = eta_yL_q.reshape((eta_yL_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][1],))
        g_x0_q = g_x0_q.reshape((g_x0_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][2],))
        g_xL_q = g_xL_q.reshape((g_xL_q.shape[0],)+(self.hparams['Q'][0],)+(self.hparams['Q'][2],))  
        u0_t0_q = u0_t0_q.reshape((u0_t0_q.shape[0],)+(self.hparams['Q'][1],)+(self.hparams['Q'][2],))       
        eta_q = torch.zeros((eta_y0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        eta_q[:,:,:,0] = eta_y0_q
        eta_q[:,:,:,-1] = eta_yL_q
        g_q = torch.zeros((g_x0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        g_q[:,:,0,:] = g_x0_q
        g_q[:,:,-1,:] = g_xL_q
        u0_t0 = torch.zeros((u0_t0_q.shape[0],)+(self.hparams['Q']), dtype=self.dtype, device=self.device)
        u0_t0[:,0,:,:] = u0_t0_q
        u0_t0_q = u0_t0
        inputfuncs = torch.stack((theta_q,f_q,eta_q,g_q,u0_t0_q), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta_q.flatten(-3,-1),f_q.flatten(-3,-1),eta_y0_q,eta_yL_q,g_x0_q,g_xL_q,u0_t0_q), dim=1)
        u_m_hat = self.systemnet.forward(inputfuncs).reshape((theta_q.shape[0],self.hparams['N']))
        return u_m_hat
    
    def DeepONet_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
        if self.timestep>0:
            u0_t0_q = opt_einsum.contract('Nn,qn->Nq', self.u_m_hat, self.basis_trial.forward(self.xi_Gamma_tT))
        theta_q = torch.tensor(theta_q, dtype=self.dtype, device=self.device)
        f_q = torch.tensor(f_q, dtype=self.dtype, device=self.device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.dtype, device=self.device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.dtype, device=self.device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.dtype, device=self.device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.dtype, device=self.device)   
        u0_t0_q = torch.tensor(u0_t0_q, dtype=self.dtype, device=self.device)  
        u_m_hat = self.DeepONet_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q).detach().cpu().numpy()
        self.u_m_hat = u_m_hat
        return u_m_hat
    
    def VarMiON_forward(self, theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q):
        eta_q = torch.zeros((eta_y0_q.shape[0],2*eta_y0_q.shape[1]), dtype=self.dtype, device=self.device)
        eta_q[:,:eta_y0_q.shape[1]] = eta_y0_q
        eta_q[:,eta_y0_q.shape[1]:] = eta_yL_q
        g_q = torch.zeros((g_x0_q.shape[0],2*g_x0_q.shape[1]), dtype=self.dtype, device=self.device)
        g_q[:,:g_x0_q.shape[1]] = g_x0_q
        g_q[:,g_x0_q.shape[1]:] = g_xL_q
        A_hat = self.systemnet.forward(theta_q)
        if self.hparams.get('Mnorm_scaling',False)==True:
            B = A_hat@self.M_lr
            B = B[:,int(A_hat.shape[1]/2):,int(A_hat.shape[2]/2):]
            scaling, v = rho_max_power_iteration(B=B, rho_0=1, x_0=self.x_0, tol=self.hparams['tol'], max_iterations=self.hparams['max_iterations'], dtype=self.dtype, device=self.device)
            scaling = torch.maximum(self.hparams['rho_max']*torch.ones_like(scaling),scaling)/self.hparams['rho_max']
            A_hat = A_hat/scaling[:,None,None]
        d_hat = self.LBranch_f.forward(f_q) + self.LBranch_eta.forward(eta_q) + self.LBranch_g.forward(g_q) + self.LBranch_u0.forward(u0_t0_q)
        u_m_hat = opt_einsum.contract('nij,nj->ni', A_hat, d_hat)
        return u_m_hat
    
    def VarMiON_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
        if self.timestep>0:
            u0_t0_q = opt_einsum.contract('Nn,qn->Nq', self.u_m_hat, self.basis_trial.forward(self.xi_Gamma_tT))
        theta_q = torch.tensor(theta_q, dtype=self.dtype, device=self.device)
        f_q = torch.tensor(f_q, dtype=self.dtype, device=self.device)
        eta_y0_q = torch.tensor(eta_y0_q, dtype=self.dtype, device=self.device)
        eta_yL_q = torch.tensor(eta_yL_q, dtype=self.dtype, device=self.device)
        g_x0_q = torch.tensor(g_x0_q, dtype=self.dtype, device=self.device)
        g_xL_q = torch.tensor(g_xL_q, dtype=self.dtype, device=self.device)   
        u0_t0_q = torch.tensor(u0_t0_q, dtype=self.dtype, device=self.device)
        u_m_hat = self.VarMiON_forward(theta_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q).detach().cpu().numpy()
        self.u_m_hat = u_m_hat
        return u_m_hat
    
    def NGO_forward(self, F, d, d_x, u_0_m, C, C_m, theta_bar):
        if self.hparams.get('scaling_equivariance',False)==True:
            F = F/theta_bar[:,None,None,None,None]
        if self.hparams.get('Neumannseries', False)==False:
            A_hat = self.systemnet.forward(F)*self.hparams['Dt']
        if self.hparams.get('Neumannseries', False)==True:
            T = torch.zeros(F.shape, dtype=self.dtype, device=self.device)
            Ti = self.Identity
            T1 = -F@self.A_0 + self.Identity
            for i in range(0, self.hparams['Neumannseries_order']):
                Ti = T1@Ti
                T = T + Ti
            A_hat = self.A_0@(self.Identity + T + self.systemnet.forward(T1))
        # if self.hparams.get('PD', False)==True:
        #     L = torch.tril(A_hat[:,0])
        #     diag_indices = torch.arange(self.hparams['N'], dtype=torch.long, device=self.device)
        #     L_diag = L[:,diag_indices,diag_indices]
        #     # L_diag_pos = torch.nn.functional.softplus(L_diag) + 1e-6
        #     L_diag_pos = torch.abs(L_diag) + 1e-6
        #     L = L.clone()  # avoid in-place on input
        #     L[:,diag_indices,diag_indices] = L_diag_pos
        #     S = L @ L.transpose(-1, -2)
        #     N = A_hat[:,1]
        #     N = 0.5*(N - N.transpose(-1, -2))
        #     A_hat = S + N
        if self.hparams.get('scaling_equivariance',False)==True:
            A_hat = A_hat/theta_bar[:,None,None]
        if self.hparams.get('Mnorm_scaling',False)==True:
            B = A_hat@self.M_lr
            B = B[:,int(A_hat.shape[1]/2):,int(A_hat.shape[2]/2):]
            M_rr22_05TBM_rr22_05_inv = self.M_rr22_05 @ B @ self.M_rr22_05_inv
            Mnorm, v = two_norm_power_iteration(B=M_rr22_05TBM_rr22_05_inv, norm2_0=self.Mnorm_0, x_0=self.x_0, tol=self.hparams['tol'], max_iterations=self.hparams['max_iterations'], dtype=self.dtype, device=self.device, model=self)
            scaling = torch.maximum(self.hparams['Mnorm_max']*torch.ones_like(Mnorm),Mnorm)/self.hparams['Mnorm_max']
            A_hat = A_hat/scaling[:,None,None]
        u_m_hat = opt_einsum.contract('nij,nj->ni', A_hat, d)
        if self.hparams['massconservation']==True:# and self.hparams['dissipation']==False:
            C_hat = opt_einsum.contract('Nn,Nn->N', u_m_hat, C_m)
            l = (C - C_hat)/opt_einsum.contract('Nn,Nn->N', C_m, C_m)
            u_m_hat = u_m_hat + l[:,None]*C_m
        # if self.hparams['dissipation']==True and self.hparams['massconservation']==False:
        #     u_m_hat = conserve_energy(self, F, d_x, u_0_m, u_m_hat)
        if self.hparams['massconservation']==True and self.hparams['dissipation']==True:
            u_m_hat = conserve_mass_and_energy(self, d_x, u_0_m, C_m, C, u_m_hat)
        return u_m_hat
    
    def NGO_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
        if self.timestep==0:
            u_0_m = torch.tensor(self.project_on_psi_tT(u0_t0_q), dtype=self.dtype, device=self.device)
        if self.timestep>0:
            u_0_m = torch.tensor(self.u_m_hat, dtype=self.dtype, device=self.device)
            u0_t0_q = opt_einsum.contract('Nn,qn->Nq', self.u_m_hat, self.basis_trial.forward(self.xi_Gamma_tT))
        F = torch.tensor(self.compute_F(theta_q, theta_x0_q, theta_xL_q), dtype=self.dtype, device=self.device)/self.hparams['Dt']
        d = torch.tensor(self.compute_d(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q), dtype=self.dtype, device=self.device)/self.hparams['Dt']
        d_x = torch.tensor(self.compute_d_x(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q), dtype=self.dtype, device=self.device)/self.hparams['Dt']
        C = torch.tensor(self.compute_C(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q), dtype=self.dtype, device=self.device)/self.hparams['Dt']
        C_m = torch.tensor(self.compute_C_m(theta_x0_q, theta_xL_q), dtype=self.dtype, device=self.device)/self.hparams['Dt']
        theta_bar = torch.tensor(opt_einsum.contract('q,Nq->N', self.w_OmegaT, theta_q), dtype=self.dtype, device=self.device)/self.hparams['Dt']
        u_m_hat = self.NGO_forward(F, d, d_x, u_0_m, C, C_m, theta_bar).detach().cpu().numpy()
        self.u_m_hat = u_m_hat
        return u_m_hat

    def FEM_forward(self, F, d):
        if self.hparams['Neumannseries']==False:
            K_inv = torch.linalg.pinv(F)
        if self.hparams['Neumannseries']==True:
            T = torch.zeros(F.shape, dtype=self.dtype, device=self.device)
            Ti = self.Identity
            T1 = -F@self.A_0 + self.Identity
            for i in range(0, self.hparams['Neumannseries_order']):
                Ti = T1@Ti
                T = T + Ti
            K_inv = self.A_0@(self.Identity + T)
        u_m_hat = opt_einsum.contract('nij,nj->ni', K_inv, d)
        return u_m_hat
    
    def FEM_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        theta_q, theta_x0_q, theta_xL_q, f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q = self.discretize_input_functions(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0)
        if self.timestep>0:
            u0_t0_q = opt_einsum.contract('Nn,qn->Nq', self.u_m_hat, self.basis_trial.forward(self.xi_Gamma_tT))
        F = torch.tensor(self.compute_F(theta_q, theta_x0_q, theta_xL_q), dtype=self.dtype, device=self.device)
        d = torch.tensor(self.compute_d(f_q, eta_y0_q, eta_yL_q, g_x0_q, g_xL_q, u0_t0_q), dtype=self.dtype, device=self.device)
        u_m_hat = self.FEM_forward(F, d).detach().cpu().numpy()
        self.u_m_hat = u_m_hat
        return u_m_hat
    
    def projection_forward(self, u_q):
        u_m_hat = self.project_input_function(u_q)
        return u_m_hat
    
    def projection_simforward(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        # u_q = discretize_functions(u, self.xi_OmegaT_scaled)
        u_q = u(self.xi_OmegaT_scaled)
        u_m_hat = self.projection_forward(u_q)
        self.u_m_hat = u_m_hat
        return u_m_hat
    
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
        u_m_hat = self.simforwardfunction(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u)
        u_q_hat = u_m_hat if self.hparams['modeltype']=='NN' else opt_einsum.contract('Nn,qn->Nq', u_m_hat, self.basis_trial.forward(x))
        torch.cuda.empty_cache()
        return u_q_hat

    def simforward_coefficients(self, theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u):
        u_m_hat = self.simforwardfunction(theta, f, eta_y0, eta_yL, g_x0, g_xL, u0, u)
        return u_m_hat

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

        if self.hparams['quadrature']=='uniform':
            quad_OmegaT = UniformQuadrature(Q=self.hparams['Q'])
            quad_Gamma_t = UniformQuadrature(Q=[self.hparams['Q'][1],self.hparams['Q'][2]])
            quad_Gamma_x = UniformQuadrature(Q=[self.hparams['Q'][0],self.hparams['Q'][2]])
            quad_Gamma_y = UniformQuadrature(Q=[self.hparams['Q'][0],self.hparams['Q'][1]])
            quad_t = UniformQuadrature(Q=[self.hparams['Q'][0]])
            quad_x = UniformQuadrature(Q=[self.hparams['Q'][1]])
            quad_y = UniformQuadrature(Q=[self.hparams['Q'][2]])

        self.w_OmegaT = quad_OmegaT.w
        self.xi_OmegaT = quad_OmegaT.xi
        
        self.w_Omega = quad_Gamma_t.w
        self.xi_Omega = quad_Gamma_t.xi

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
        if self.hparams['quadrature_L']=='uniform':
            quad_OmegaT_L = UniformQuadrature(Q=self.hparams['Q_L'])
        self.w_OmegaT_L = quad_OmegaT_L.w
        self.xi_OmegaT_L = quad_OmegaT_L.xi

    def rescale_quadrature_t(self):
        self.w_OmegaT = self.w_OmegaT*self.hparams['Dt']
        self.w_OmegaT_L = self.w_OmegaT_L*self.hparams['Dt']
        self.w_Gamma_x0 = self.w_Gamma_x0*self.hparams['Dt']
        self.w_Gamma_xL = self.w_Gamma_xL*self.hparams['Dt']
        self.w_Gamma_y0 = self.w_Gamma_y0*self.hparams['Dt']
        self.w_Gamma_yL = self.w_Gamma_yL*self.hparams['Dt']
        self.w_t = self.w_t*self.hparams['Dt']

        self.xi_OmegaT_scaled = np.copy(self.xi_OmegaT)
        self.xi_OmegaT_L_scaled = np.copy(self.xi_OmegaT_L)
        self.xi_Gamma_t0_scaled = np.copy(self.xi_Gamma_t0)
        self.xi_Gamma_tT_scaled = np.copy(self.xi_Gamma_tT)
        self.xi_Gamma_x0_scaled = np.copy(self.xi_Gamma_x0)
        self.xi_Gamma_xL_scaled = np.copy(self.xi_Gamma_xL)
        self.xi_Gamma_y0_scaled = np.copy(self.xi_Gamma_y0)
        self.xi_Gamma_yL_scaled = np.copy(self.xi_Gamma_yL)
        self.xi_OmegaT_scaled[:,0] = self.xi_OmegaT[:,0]*self.hparams['Dt']
        self.xi_OmegaT_L_scaled[:,0] = self.xi_OmegaT_L[:,0]*self.hparams['Dt']
        self.xi_Gamma_t0_scaled[:,0] = self.xi_Gamma_t0[:,0]*self.hparams['Dt']
        self.xi_Gamma_tT_scaled[:,0] = self.xi_Gamma_tT[:,0]*self.hparams['Dt']
        self.xi_Gamma_x0_scaled[:,0] = self.xi_Gamma_x0[:,0]*self.hparams['Dt']
        self.xi_Gamma_xL_scaled[:,0] = self.xi_Gamma_xL[:,0]*self.hparams['Dt']
        self.xi_Gamma_y0_scaled[:,0] = self.xi_Gamma_y0[:,0]*self.hparams['Dt']
        self.xi_Gamma_yL_scaled[:,0] = self.xi_Gamma_yL[:,0]*self.hparams['Dt']

    def translate_quadrature_t(self):
        self.xi_OmegaT_scaled[:,0] += self.hparams['Dt']
        self.xi_Gamma_t0_scaled[:,0] += self.hparams['Dt']
        self.xi_Gamma_tT_scaled[:,0] += self.hparams['Dt']
        self.xi_Gamma_x0_scaled[:,0] += self.hparams['Dt']
        self.xi_Gamma_xL_scaled[:,0] += self.hparams['Dt']
        self.xi_Gamma_y0_scaled[:,0] += self.hparams['Dt']
        self.xi_Gamma_yL_scaled[:,0] += self.hparams['Dt']

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
                    # loss += self.hparams['solution_loss'](self.M, u_hat, u)
                if self.hparams['output_coefficients']==False:
                    loss += self.hparams['solution_loss'](self.w_OmegaT_L, u_hat, u)
                    # loss += self.hparams['solution_loss'](self.w_Gamma_tT, u_hat, u)
            if self.hparams.get('matrix_loss',None)!=None:
                F = inputs[0]
                if self.hparams.get('Neumannseries', False)==False:
                    A_hat = self.systemnet.forward(F)
                if self.hparams.get('Neumannseries', False)==True:
                    A_0 = self.A_0 if self.hparams.get('A0net')==None else self.A0net.systemnet.forward(F)
                    T = torch.zeros(F.shape, dtype=self.dtype, device=self.device)
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
                # loss += self.hparams['solution_loss'](self.M, u_hat, u)
            if self.hparams['output_coefficients']==False:
                loss += self.hparams['solution_loss'](self.w_OmegaT_L, u_hat, u)
                # loss += self.hparams['solution_loss'](self.w_Gamma_tT, u_hat, u)
        if self.hparams.get('matrix_loss',None)!=None:
            F = inputs[0]
            if self.hparams.get('Neumannseries',False)==False:
                A_hat = self.systemnet.forward(F)
            if self.hparams.get('Neumannseries', False)==True:
                A_0 = self.A_0 if self.hparams.get('A0net')==None else self.A0net.systemnet.forward(F)
                T = torch.zeros(F.shape, dtype=self.dtype, device=self.device)
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
            #   metric = self.hparams['solution_loss'](self.M, u_hat, u)
        if self.hparams['output_coefficients']==False:
            metric = self.hparams['metric'](self.w_OmegaT_L, u_hat, u)
            # metric = self.hparams['metric'](self.w_Gamma_tT, u_hat, u)
        self.metric = metric.item()
        self.log('val_loss', loss)
        self.log('metric', metric)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['hparams'] = self.hparams
        
    def on_fit_start(self):
        # self.device = self.hparams
        torch.set_num_threads(2)
        self.metric_best = 100
        if self.hparams['modeltype']!='NN':
            self.M_lr = torch.tensor(self.M_lr, dtype=self.dtype, device=self.device)
            self.M_rr = torch.tensor(self.M_rr, dtype=self.dtype, device=self.device)
            self.M_rr22 = torch.tensor(self.M_rr22, dtype=self.dtype, device=self.device)
            self.M_rr22_05 = torch.tensor(self.M_rr22_05, dtype=self.dtype, device=self.device)
            self.M_rr22_05_inv = torch.tensor(self.M_rr22_05_inv, dtype=self.dtype, device=self.device)

            if self.hparams['output_coefficients']==False:
                # self.psix = torch.tensor(self.basis_trial.forward(self.xi_Gamma_tT), dtype=self.dtype, device=self.device)
                self.psix = torch.tensor(self.basis_trial.forward(self.xi_OmegaT_L), dtype=self.dtype, device=self.device)
                self.psix = self.psix.to(self.device)
            if self.hparams['output_coefficients']==True:
                self.M_phiphi = torch.tensor(self.M_phiphi, dtype=self.dtype, device=self.device).to(self.device)
        self.w_OmegaT_L = torch.tensor(self.w_OmegaT_L, dtype=self.dtype, device=self.device).to(self.device)
        # self.w_Gamma_tT = torch.tensor(self.w_Gamma_tT, dtype=self.dtype, device=self.device).to(self.device)
        self.systemnet.device = self.device
        if self.hparams.get('A0net')!=None:
            self.A0net = loadmodelfromlabel(model=NeuralOperator, label=self.hparams['A0net'][1], logdir='../../../nnlogs', sublogdir=self.hparams['A0net'][0], map_location=self.hparams['device'])
            self.A0net.systemnet = self.A0net.systemnet.to(self.device)
            for param in self.A0net.parameters():
                param.requires_grad = False

    def on_validation_epoch_end(self):
        if self.metric<self.metric_best:
            self.metric_best = self.metric
            # if self.hparams['Mnorm_scaling']==True:
            #     # with torch.no_grad():
            #     A_hat = self.systemnet.forward(self.F_0)
            #     B = A_hat@self.M_lr
            #     B = B[:,int(A_hat.shape[1]/2):,int(A_hat.shape[2]/2):]
            #     rho_0, self.x_0 = rho_max_power_iteration(B=B, rho_0=1, x_0=self.x_0, tol=self.hparams['tol'], max_iterations=self.hparams['max_iterations'], dtype=self.dtype, device=self.device)
            #     print('Lowest error, saving eigenvector estimate:')
            #     # print(self.x_0)
            #     torch.save(self.x_0.cpu(), self.hparams['logdir']+'/x_0.pt')
            if self.hparams.get('Mnorm_scaling',False)==True:
                # with torch.no_grad():
                A_hat = self.systemnet.forward(self.F_0)
                B = A_hat@self.M_lr
                B = B[:,int(A_hat.shape[1]/2):,int(A_hat.shape[2]/2):]
                C = self.M_rr22_05 @ B @ self.M_rr22_05_inv
                self.C_0 = C
                Mnorm_0, self.x_0 = two_norm_power_iteration(B=C, norm2_0=self.Mnorm_0, x_0=self.x_0, tol=self.hparams['tol'], max_iterations=self.hparams['max_iterations'], dtype=self.dtype, device=self.device, model=self)
                print('Lowest error, saving eigenvector estimate:')
                # print(self.x_0)
                torch.save(self.x_0.cpu(), self.hparams['logdir']+'/x_0.pt')
        print(self.metric)
        if self.hparams['switch_threshold']!=None:
            if self.metric[-1]<self.hparams['switch_threshold']:
                    self.optimizer_idx = 1
                    self.hparams['batch_size'] = self.hparams['N_samples']