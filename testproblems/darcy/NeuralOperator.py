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
            self.basis_test = BSplineInterpolatedPOD2D(N_samples=self.hparams['N_samples_train'], variables=hparams['variables'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'], w=self.w_Omega, xi=self.xi_Omega, N=self.hparams['N'], device=self.used_device)
            self.basis_trial = BSplineInterpolatedPOD2D(N_samples=self.hparams['N_samples_train'], variables=hparams['variables'], l_min=self.hparams['l_min'], l_max=self.hparams['l_max'], w=self.w_Omega, xi=self.xi_Omega, N=self.hparams['N'], device=self.used_device)
        #Basis evaluation at quadrature points
        self.psix = torch.tensor(self.basis_trial.forward(self.xi_Omega_L.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        #A_0 (K inverse for constant theta)
        # if self.hparams['Neumannseries']==True:
        self.F_0, self.A_0 = self.compute_F_0_A_0()
        #Identity
        self.Identity = torch.eye(self.hparams['N'], dtype=self.hparams['dtype'], device=self.used_device)
        self = self.to(self.hparams['dtype'])

    def discretize_input_functions(self, theta, f, etab, etat, gl, gr):
        theta_d = discretize_functions(theta, self.xi_Omega, dtype=self.hparams['dtype'], device=self.used_device)
        theta_g_d = discretize_functions(theta, self.xi_Gamma_g, dtype=self.hparams['dtype'], device=self.used_device)
        f_d = discretize_functions(f, self.xi_Omega, dtype=self.hparams['dtype'], device=self.used_device)
        etab_d = discretize_functions(etab, self.xi_Gamma_b, dtype=self.hparams['dtype'], device=self.used_device)
        etat_d = discretize_functions(etat, self.xi_Gamma_t, dtype=self.hparams['dtype'], device=self.used_device)
        gl_d = discretize_functions(gl, self.xi_Gamma_l, dtype=self.hparams['dtype'], device=self.used_device)
        gr_d = discretize_functions(gr, self.xi_Gamma_r, dtype=self.hparams['dtype'], device=self.used_device)
        return theta_d, theta_g_d, f_d, etab_d, etat_d, gl_d, gr_d
    
    def discretize_output_function(self, u):
        u_d = discretize_functions(u, self.xi_Omega_L, dtype=self.hparams['dtype'], device=self.used_device)
        return u_d

    def compute_F(self, theta, theta_g):
        if  self.hparams['modeltype']=='data NGO':
            basis_test = torch.tensor(self.basis_test.forward(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
            F = torch.zeros((theta.shape[0],self.hparams['N']), dtype=self.hparams['dtype'], device=self.used_device)
            if self.hparams.get('Assembly_mode','parallel')=='parallel':
                F = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, theta)
            if self.hparams.get('Assembly_mode','parallel')=='series':
                for i in range(theta.shape[0]):
                    F[i] = opt_einsum.contract('q,qm,q->m', self.w_Omega, basis_test, theta[i])
            F = F.reshape((theta.shape[0],self.hparams['h'][0],self.hparams['h'][1]))
        if  self.hparams['modeltype']=='matrix data NGO':
            basis_test = torch.tensor(self.basis_test.forward(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
            basis_trial = torch.tensor(self.basis_trial.forward(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
            F = opt_einsum.contract('q,q,qm,qn->mn', self.w_Omega, theta, basis_test, basis_trial)
        if  self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM' or self.hparams['modeltype']=='projection':
            gradbasis_test = torch.tensor(self.basis_test.grad(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
            gradbasis_trial = torch.tensor(self.basis_trial.grad(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
            basis_test_g = torch.tensor(self.basis_test.forward(self.xi_Gamma_g.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
            gradbasis_test_g = torch.tensor(self.basis_test.grad(self.xi_Gamma_g.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
            basis_trial_g = torch.tensor(self.basis_trial.forward(self.xi_Gamma_g.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
            gradbasis_trial_g = torch.tensor(self.basis_trial.grad(self.xi_Gamma_g.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
            F = torch.zeros((theta.shape[0],self.hparams['N'],self.hparams['N']), dtype=self.hparams['dtype'], device=self.used_device)
            if self.hparams.get('Assembly_mode','parallel')=='parallel':
                F = opt_einsum.contract('q,Nq,qmx,qnx->Nmn', self.w_Omega, theta, gradbasis_test, gradbasis_trial)
                F += -opt_einsum.contract('q,qm,qx,Nq,qnx->Nmn', self.w_Gamma_g, basis_test_g, self.n_Gamma_g, theta_g, gradbasis_trial_g)
                F += -opt_einsum.contract('q,qn,qx,Nq,qmx->Nmn', self.w_Gamma_g, basis_trial_g, self.n_Gamma_g, theta_g, gradbasis_test_g)
            if self.hparams.get('gamma_stabilization',0)!=0:
                F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_g, basis_test_g, theta_g, basis_trial_g)
            if self.hparams.get('Assembly_mode','parallel')=='series':
                for i in range(theta.shape[0]):
                    print(i)
                    F[i] = opt_einsum.contract('q,q,qmx,qnx->mn', self.w_Omega, theta[i], gradbasis_test, gradbasis_trial)
                    F[i] += -opt_einsum.contract('q,qm,qx,q,qnx->mn', self.w_Gamma_g, basis_test_g, self.n_Gamma_g, theta_g[i], gradbasis_trial_g)
                    F[i] += -opt_einsum.contract('q,qn,qx,q,qmx->mn', self.w_Gamma_g, basis_trial_g, self.n_Gamma_g, theta_g[i], gradbasis_test_g)
                if self.hparams.get('gamma_stabilization',0)!=0:
                    F[i] += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,q,qn->mn', self.w_Gamma_g, basis_test_g, theta_g[i], basis_trial_g)
        return F
    
    def compute_F_0_A_0(self):
        theta = torch.ones(self.w_Omega.shape, dtype=self.hparams['dtype'], device=self.used_device)
        theta_g = torch.ones(self.w_Gamma_g.shape, dtype=self.hparams['dtype'], device=self.used_device)
        gradbasis_test = torch.tensor(self.basis_test.grad(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        gradbasis_trial = torch.tensor(self.basis_trial.grad(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        basis_test_g = torch.tensor(self.basis_test.forward(self.xi_Gamma_g.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        gradbasis_test_g = torch.tensor(self.basis_test.grad(self.xi_Gamma_g.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        basis_trial_g = torch.tensor(self.basis_trial.forward(self.xi_Gamma_g.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        gradbasis_trial_g = torch.tensor(self.basis_trial.grad(self.xi_Gamma_g.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        F_0 = torch.zeros((theta.shape[0],self.hparams['N'],self.hparams['N']), dtype=self.hparams['dtype'], device=self.used_device)
        F_0 = opt_einsum.contract('q,q,qmx,qnx->mn', self.w_Omega, theta, gradbasis_test, gradbasis_trial)
        F_0 += -opt_einsum.contract('q,qm,qx,q,qnx->mn', self.w_Gamma_g, basis_test_g, self.n_Gamma_g, theta_g, gradbasis_trial_g)
        F_0 += -opt_einsum.contract('q,qn,qx,q,qmx->mn', self.w_Gamma_g, basis_trial_g, self.n_Gamma_g, theta_g, gradbasis_test_g)
        if self.hparams.get('gamma_stabilization',None)!=None:
            F_0 += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,q,qn->mn', self.w_Gamma_g, basis_test_g, theta_g, basis_trial_g)
        A_0 = torch.linalg.pinv(F_0)
        A_0 = torch.tensor(A_0, dtype=self.hparams['dtype'], device=self.used_device)
        if  self.hparams['modeltype']=='data NGO':
            basis_test = torch.tensor(self.basis_test.forward(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
            F = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, theta).reshape((theta.shape[0],self.hparams['h'][0],self.hparams['h'][1]))
            F_0 = torch.tensor(F, dtype=self.hparams['dtype'], device=self.used_device)
        return F_0, A_0
    
    def compute_d(self, f, etab, etat, gl, gr):
        basis_test = torch.tensor(self.basis_test.forward(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        basis_test_b = torch.tensor(self.basis_test.forward(self.xi_Gamma_b.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        basis_test_t = torch.tensor(self.basis_test.forward(self.xi_Gamma_t.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        basis_test_l = torch.tensor(self.basis_test.forward(self.xi_Gamma_l.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        basis_test_r = torch.tensor(self.basis_test.forward(self.xi_Gamma_r.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        gradbasis_test_l = torch.tensor(self.basis_test.grad(self.xi_Gamma_l.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        gradbasis_test_r = torch.tensor(self.basis_test.grad(self.xi_Gamma_r.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        d = torch.zeros((f.shape[0],self.hparams['N']), dtype=self.hparams['dtype'], device=self.used_device)
        if self.hparams.get('Assembly_mode','parallel')=='parallel':
            d = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, f)
            d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_b, basis_test_b, etab)
            d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_t, basis_test_t, etat)
            d -= opt_einsum.contract('q,qx,qmx,Nq->Nm', self.w_Gamma_l, self.n_l, gradbasis_test_l, gl)
            d -= opt_einsum.contract('q,qx,qmx,Nq->Nm', self.w_Gamma_r, self.n_r, gradbasis_test_r, gr)
            if self.hparams.get('gamma_stabilization',0)!=0:
                d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_l, basis_test_l, gl)
                d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_r, basis_test_r, gr)          
        if self.hparams.get('Assembly_mode','parallel')=='series':
            for i in range(f.shape[0]):
                print(i)
                d[i] = opt_einsum.contract('q,qm,q->m', self.w_Omega, basis_test, f[i])
                d[i] += opt_einsum.contract('q,qm,q->m', self.w_Gamma_b, basis_test_b, etab[i])
                d[i] += opt_einsum.contract('q,qm,q->m', self.w_Gamma_t, basis_test_t, etat[i])
                d[i] -= opt_einsum.contract('q,qx,qmx,q->m', self.w_Gamma_l, self.n_l, gradbasis_test_l, gl[i])
                d[i] -= opt_einsum.contract('q,qx,qmx,q->m', self.w_Gamma_r, self.n_r, gradbasis_test_r, gr[i])
                if self.hparams.get('gamma_stabilization',0)!=0:
                    d[i] += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,q->m', self.w_Gamma_l, basis_test_l, gl[i])
                    d[i] += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,q->m', self.w_Gamma_r, basis_test_r, gr[i])        
        return d
    
    def NN_forward(self, theta, f, etab, etat, gl, gr):
        theta = theta.reshape((theta.shape[0],self.hparams['Q'],self.hparams['Q']))
        f = f.reshape((f.shape[0],self.hparams['Q'],self.hparams['Q']))
        eta = torch.zeros((etab.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        eta[:,:,0] = etab
        eta[:,:,-1] = etat
        g = torch.zeros((gl.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        g[:,0,:] = gl
        g[:,-1,:] = gr
        inputfuncs = torch.stack((theta,f,eta,g), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta.flatten(-2,-1),f.flatten(-2,-1),etab,etat,gl,gr), dim=1)
        u_hat = self.systemnet.forward(inputfuncs).reshape((theta.shape[0],self.hparams['Q_L']**self.hparams['d']))
        return u_hat
    
    def DeepONet_forward(self, theta, f, etab, etat, gl, gr):
        theta = theta.reshape((theta.shape[0],self.hparams['Q'],self.hparams['Q']))
        f = f.reshape((f.shape[0],self.hparams['Q'],self.hparams['Q']))
        eta = torch.zeros((etab.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        eta[:,:,0] = etab
        eta[:,:,-1] = etat
        g = torch.zeros((gl.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        g[:,0,:] = gl
        g[:,-1,:] = gr
        inputfuncs = torch.stack((theta,f,eta,g), dim=1)
        if self.hparams['systemnet']==MLP:
            inputfuncs = torch.cat((theta.flatten(-2,-1),f.flatten(-2,-1),etab,etat,gl,gr), dim=1)
        u_n = self.systemnet.forward(inputfuncs).reshape((theta.shape[0],self.hparams['N']))
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat
    
    def VarMiON_forward(self, theta, f, etab, etat, gl, gr):
        eta = torch.zeros((etab.shape[0],2*etab.shape[1]), dtype=self.hparams['dtype'], device=self.used_device)
        eta[:,:etab.shape[1]] = etab
        eta[:,etab.shape[1]:] = etat
        g = torch.zeros((gl.shape[0],2*gl.shape[1]), dtype=self.hparams['dtype'], device=self.used_device)
        g[:,:gl.shape[1]] = gl
        g[:,gl.shape[1]:] = gr
        systemnet = self.systemnet.forward(theta)
        LBranch = self.LBranch_f.forward(f) + self.LBranch_eta.forward(eta) + self.LBranch_g.forward(g)
        u_n = opt_einsum.contract('nij,nj->ni', systemnet, LBranch)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat

    def NGO_forward(self, scaling, F, d):
        if self.hparams.get('scaling_equivariance',False)==True:
            F = F/scaling[:,None,None]
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
            A = A/scaling[:,None,None]        
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
        u_q = torch.tensor(discretize_functions(u, self.xi_Omega, dtype=self.hparams['dtype'], device=self.used_device), dtype=self.hparams['dtype'])
        basis_test = torch.tensor(self.basis_test.forward(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        basis_trial = torch.tensor(self.basis_trial.forward(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        u_w = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, u_q)
        M = opt_einsum.contract('q,qm,qn->mn', self.w_Omega, basis_test, basis_trial)
        M_inv = torch.linalg.pinv(M)
        u_n = opt_einsum.contract('mn,Nm->Nn', M_inv, u_w)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat

    def Greens_function(self, theta, x, xp):
        theta_d = discretize_functions(theta, self.xi_Omega, dtype=self.hparams['dtype'], device=self.used_device)
        theta_g_d = discretize_functions(theta, self.xi_Gamma_g, dtype=self.hparams['dtype'], device=self.used_device)
        F = torch.tensor(self.compute_F(theta_d, theta_g_d), dtype=self.hparams['dtype'], device=self.used_device)
        if self.hparams['modeltype']=='FEM':
            if self.hparams.get('Neumannseries', False)==False:
                A = torch.linalg.pinv(F)
            if self.hparams.get('Neumannseries', False)==True:
                T = torch.zeros(F.shape, dtype=self.hparams['dtype'], device=self.used_device)
                Ti = self.Identity
                T1 = -F@self.A_0 + self.Identity
                for i in range(0, self.hparams['Neumannseries_order']):
                    Ti = T1@Ti
                    T = T + Ti
                A = self.A_0@(self.Identity + T)
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO':
            A = self.systemnet.forward(F)
        psi = torch.tensor(self.basis_trial.forward(x.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        phi = torch.tensor(self.basis_test.forward(xp.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        G = opt_einsum.contract('qm,Nmn,qn->Nq', phi, A, psi)
        return G
    
    def compute_projection_coeffs(self, u):
        u_q = torch.tensor(discretize_functions(u, self.xi_Omega, dtype=self.hparams['dtype'], device=self.used_device), dtype=self.hparams['dtype'])
        basis_test = torch.tensor(self.basis_test.forward(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        basis_trial = torch.tensor(self.basis_trial.forward(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        u_w = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, u_q)
        M = opt_einsum.contract('q,qm,qn->mn', self.w_Omega, basis_test, basis_trial)
        M_inv = torch.linalg.pinv(M)
        u_n = opt_einsum.contract('mn,Nm->Nn', M_inv, u_w)
        return u_n

    def Neumann_criterion(self, theta):
        theta_d = discretize_functions(theta, self.xi_Omega, dtype=self.hparams['dtype'], device=self.used_device)
        theta_g_d = discretize_functions(theta, self.xi_Gamma_g, dtype=self.hparams['dtype'], device=self.used_device)
        F = torch.tensor(self.compute_F(theta_d, theta_g_d), dtype=self.hparams['dtype'], device=self.used_device)
        if self.hparams.get('scaling_equivariance',False)==True:
            scaling = torch.tensor(torch.sum(self.w_Omega[None,:]*theta_d, axis=-1), dtype=self.hparams['dtype'], device=self.used_device)
            F = F/scaling[:,None,None]
        T1 = -F@self.A_0 + self.Identity
        eigvals = torch.linalg.eigvals(T1)
        spectralradius = torch.amax(torch.real((eigvals*torch.conj(eigvals))**(1/2)), dim=-1)
        return spectralradius
    
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
    
    def simforward(self, theta_in, f, etab, etat, gl, gr, x, u):
        self.geometry()
        #self.compute_F_0_A_0()
        theta, theta_g, f, etab, etat, gl, gr = self.discretize_input_functions(theta_in, f, etab, etat, gl, gr)
        if self.hparams.get('project_inputs',False)==True:
            theta_n = self.compute_projection_coeffs(theta_in)
            psi_Omega = torch.tensor(self.basis_trial.forward(self.xi_Omega.cpu().numpy()), dtype=self.hparams['dtype'], device=self.hparams['assembly_device'])
            theta = opt_einsum.contract('Nn,qn->Nq', theta_n, psi_Omega)
            psi_Gamma_g = torch.tensor(self.basis_trial.forward(self.xi_Gamma_g.cpu().numpy()), dtype=self.hparams['dtype'], device=self.hparams['assembly_device'])
            theta_g = opt_einsum.contract('Nn,qn->Nq', theta_n, psi_Gamma_g)
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO' or self.hparams['modeltype']=='FEM':
            F = torch.tensor(self.compute_F(theta, theta_g), dtype=self.hparams['dtype'], device=self.used_device)
            d = torch.tensor(self.compute_d(f, etab, etat, gl, gr), dtype=self.hparams['dtype'], device=self.used_device)
        theta = torch.tensor(theta, dtype=self.hparams['dtype'], device=self.used_device)
        f = torch.tensor(f, dtype=self.hparams['dtype'], device=self.used_device)
        etab = torch.tensor(etab, dtype=self.hparams['dtype'], device=self.used_device)
        etat = torch.tensor(etat, dtype=self.hparams['dtype'], device=self.used_device)
        gl = torch.tensor(gl, dtype=self.hparams['dtype'], device=self.used_device)
        gr = torch.tensor(gr, dtype=self.hparams['dtype'], device=self.used_device)    
        self.psix = torch.tensor(self.basis_trial.forward(x.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)
        if self.hparams['modeltype']=='NN':
            u_hat = self.NN_forward(theta, f, etab, etat, gl, gr)
        if self.hparams['modeltype']=='DeepONet':
            u_hat = self.DeepONet_forward(theta, f, etab, etat, gl, gr)
        if self.hparams['modeltype']=='VarMiON':
            u_hat = self.VarMiON_forward(theta, f, etab, etat, gl, gr)
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO':
            scaling = torch.tensor(torch.sum(self.w_Omega[None,:]*theta, axis=-1), dtype=self.hparams['dtype'], device=self.used_device)
            # lambdas = torch.linalg.eigvals(F)
            # lambdas_abs = torch.real((lambdas*torch.conj(lambdas))**(1/2))
            # lambda_max = torch.amax(lambdas_abs, axis=-1)
            # scaling = lambda_max
            u_hat = self.NGO_forward(scaling, F, d)
        if self.hparams['modeltype']=='FEM':
            u_hat = self.FEM_forward(F, d)
        if self.hparams['modeltype']=='projection':
            u_hat = self.projection_forward(u)
        return u_hat
    
    def geometry(self):
        #Quadrature
        if self.hparams['quadrature']=='uniform':
            quadrature = UniformQuadrature2D(Q=self.hparams['Q'])
        if self.hparams['quadrature']=='Gauss-Legendre':
            quadrature = GaussLegendreQuadrature2D(Q=self.hparams['Q'], n_elements=self.hparams['n_elements'])
        self.xi_Omega = torch.tensor(quadrature.xi_Omega, dtype=self.hparams['dtype'], device=self.used_device)
        self.xi_Gamma_b = torch.tensor(quadrature.xi_Gamma_b, dtype=self.hparams['dtype'], device=self.used_device)
        self.xi_Gamma_t = torch.tensor(quadrature.xi_Gamma_t, dtype=self.hparams['dtype'], device=self.used_device)
        self.xi_Gamma_l = torch.tensor(quadrature.xi_Gamma_l, dtype=self.hparams['dtype'], device=self.used_device)
        self.xi_Gamma_r = torch.tensor(quadrature.xi_Gamma_r, dtype=self.hparams['dtype'], device=self.used_device)
        self.xi_Gamma_eta = torch.tensor(quadrature.xi_Gamma_eta, dtype=self.hparams['dtype'], device=self.used_device)
        self.xi_Gamma_g = torch.tensor(quadrature.xi_Gamma_g, dtype=self.hparams['dtype'], device=self.used_device)
        self.w_Omega = torch.tensor(quadrature.w_Omega, dtype=self.hparams['dtype'], device=self.used_device)
        self.w_Gamma_b = torch.tensor(quadrature.w_Gamma_b, dtype=self.hparams['dtype'], device=self.used_device)
        self.w_Gamma_t = torch.tensor(quadrature.w_Gamma_t, dtype=self.hparams['dtype'], device=self.used_device)
        self.w_Gamma_l = torch.tensor(quadrature.w_Gamma_l, dtype=self.hparams['dtype'], device=self.used_device)
        self.w_Gamma_r =torch.tensor( quadrature.w_Gamma_r, dtype=self.hparams['dtype'], device=self.used_device)
        self.w_Gamma_eta = torch.tensor(quadrature.w_Gamma_eta, dtype=self.hparams['dtype'], device=self.used_device)
        self.w_Gamma_g = torch.tensor(quadrature.w_Gamma_g, dtype=self.hparams['dtype'], device=self.used_device)

        #Outward normal
        # if self.hparams['Q'][0]!=self.hparams['Q'][1]:
        #     raise ValueError
        outwardnormal = UnitSquareOutwardNormal(Q=self.hparams['Q'])
        self.n_b = torch.tensor(outwardnormal.n_b, dtype=self.hparams['dtype'], device=self.used_device)
        self.n_t = torch.tensor(outwardnormal.n_t, dtype=self.hparams['dtype'], device=self.used_device)
        self.n_l = torch.tensor(outwardnormal.n_l, dtype=self.hparams['dtype'], device=self.used_device)
        self.n_r = torch.tensor(outwardnormal.n_r, dtype=self.hparams['dtype'], device=self.used_device)
        self.n_Gamma_eta = torch.tensor(outwardnormal.n_Gamma_eta, dtype=self.hparams['dtype'], device=self.used_device)
        self.n_Gamma_g = torch.tensor(outwardnormal.n_Gamma_g, dtype=self.hparams['dtype'], device=self.used_device)
        #Loss quadrature
        if self.hparams['quadrature_L']=='uniform':
            quadrature_L = UniformQuadrature2D(Q=self.hparams['Q_L'])
        if self.hparams['quadrature_L']=='Gauss-Legendre':
            quadrature_L = GaussLegendreQuadrature2D(Q=self.hparams['Q_L'], n_elements=self.hparams['n_elements_L'])
        self.xi_Omega_L = torch.tensor(quadrature_L.xi_Omega, dtype=self.hparams['dtype'], device=self.used_device)
        self.w_Omega_L = torch.tensor(quadrature_L.w_Omega, dtype=self.hparams['dtype'], device=self.used_device)

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
                loss += self.hparams['solution_loss'](self.w_Omega_L, u_hat, u)
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
            loss += self.hparams['solution_loss'](self.w_Omega_L, u_hat, u)
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
        metric = self.hparams['metric'](self.w_Omega_L, u_hat, u)
        self.metric.append(metric)
        self.log('val_loss', loss)
        self.log('metric', metric)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['hparams'] = self.hparams
        
    def on_fit_start(self):
        self.used_device = self.used_device
        torch.set_num_threads(2)
        self.psix = self.psix.to(self.used_device)
        self.w_Omega_L = torch.tensor(self.w_Omega_L).to(self.used_device)
        self.systemnet.device = self.used_device
        if self.hparams.get('A0net')!=None:
            self.A0net = loadmodelfromlabel(model=NeuralOperator, label=self.hparams['A0net'][1], logdir='../../../nnlogs', sublogdir=self.hparams['A0net'][0], map_location=self.hparams['device'])
            self.A0net.systemnet = self.A0net.systemnet.to(self.used_device)
            for param in self.A0net.parameters():
                param.requires_grad = False

    def on_validation_epoch_end(self):
        print(self.metric[-1])
        if self.hparams['switch_threshold']!=None:
            if self.metric[-1]<self.hparams['switch_threshold']:
                    self.optimizer_idx = 1
                    self.hparams['batch_size'] = self.hparams['N_samples']