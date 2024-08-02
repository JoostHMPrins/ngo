import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
import numpy as np
import opt_einsum
import time
from numba.experimental import jitclass
from numba import float32, int32
import numba
from typing import List, Callable

import sys
sys.path.insert(0, '../../ml')
from systemnets import *
from basisfunctions import *
from quadrature import *
from customlayers import *
from customlosses import *

class NGO(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hparams.update(params['hparams'])
        #Model type
        self.init_modeltype()
        #System net
        self.systemnet = self.hparams['systemnet'](params)
        if self.hparams.get('modeltype',False)=='VarMiON':
            self.LBranch_f = LBranchNet(params, input_dim=self.hparams['Q']**params['simparams']['d'], output_dim=self.hparams['N'])
            self.LBranch_eta = LBranchNet(params, input_dim=2*self.hparams['Q'], output_dim=self.hparams['N'])
            self.LBranch_g = LBranchNet(params, input_dim=2*self.hparams['Q'], output_dim=self.hparams['N'])
        #Bases
        self.basis_test = TensorizedBasis(self.hparams['test_bases'])
        self.basis_trial = TensorizedBasis(self.hparams['trial_bases'])
        #Geometry and quadrature
        self.geometry()
        self = self.to(self.hparams['dtype'])
    
    def discretize_input_functions(self, theta_f, f_f, etab_f, etat_f, gl_f, gr_f):
        theta = []
        theta_g = []
        f = []
        etab = []
        etat = []
        gl = []
        gr = []
        for i in range(len(theta_f)):
            theta.append(theta_f[i](self.xi_Omega))
            theta_g.append(theta_f[i](self.xi_Gamma_g))
            f.append(f_f[i](self.xi_Omega))
            etab.append(etab_f[i](self.xi_Gamma_b))
            etat.append(etat_f[i](self.xi_Gamma_t))
            gl.append(gl_f[i](self.xi_Gamma_l))
            gr.append(gr_f[i](self.xi_Gamma_r))
        return np.array(theta), np.array(theta_g), np.array(f), np.array(etab), np.array(etat), np.array(gl), np.array(gr)
    
    def discretize_output_function(self, u_f):
        u = []
        for i in range(len(u_f)):
            u.append(u_f[i](self.xi_Omega_L))
        return np.array(u)

    def compute_F(self, theta, theta_g):
        if self.hparams.get('model/data',False)=='data':
            basis_test = self.basis_test.forward(self.xi_Omega)
            F = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, theta).reshape((theta.shape[0],self.hparams['h'][0],self.hparams['h'][1]))
        if self.hparams.get('model/data',False)=='matrix data':
            basis_test = self.basis_test.forward(self.xi_Omega)
            basis_trial = self.basis_trial.forward(self.xi_Omega)
            F = opt_einsum.contract('q,Nq,qm,qn->Nmn', self.w_Omega, theta, basis_test, basis_trial)
        if self.hparams.get('model/data',False)=='model':
            gradbasis_test = self.basis_test.grad(self.xi_Omega)
            gradbasis_trial = self.basis_trial.grad(self.xi_Omega)
            basis_test_g = self.basis_test.forward(self.xi_Gamma_g)
            gradbasis_test_g = self.basis_test.grad(self.xi_Gamma_g)
            basis_trial_g = self.basis_trial.forward(self.xi_Gamma_g)
            gradbasis_trial_g = self.basis_trial.grad(self.xi_Gamma_g)
            F = opt_einsum.contract('q,Nq,qmx,qnx->Nmn', self.w_Omega, theta, gradbasis_test, gradbasis_trial)
            F += -opt_einsum.contract('q,qm,qx,Nq,qnx->Nmn', self.w_Gamma_g, basis_test_g, self.n_Gamma_g, theta_g, gradbasis_trial_g)
            F += -opt_einsum.contract('q,qn,qx,Nq,qmx->Nmn', self.w_Gamma_g, basis_trial_g, self.n_Gamma_g, theta_g, gradbasis_test_g)
        if self.hparams.get('gamma_stabilization',0)!=0:
            F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_g, basis_test_g, theta_g, basis_trial_g)
        return F
    
    def compute_d(self, f, etab, etat, gl, gr):
        basis_test = self.basis_test.forward(self.xi_Omega)
        basis_test_b = self.basis_test.forward(self.xi_Gamma_b)
        basis_test_t = self.basis_test.forward(self.xi_Gamma_t)
        basis_test_l = self.basis_test.forward(self.xi_Gamma_l)
        basis_test_r = self.basis_test.forward(self.xi_Gamma_r)
        gradbasis_test_l = self.basis_test.grad(self.xi_Gamma_l)
        gradbasis_test_r = self.basis_test.grad(self.xi_Gamma_r)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, f)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_b, basis_test_b, etab)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_t, basis_test_t, etat)
        d -= opt_einsum.contract('q,qx,qmx,Nq->Nm', self.w_Gamma_l, self.n_l, gradbasis_test_l, gl)
        d -= opt_einsum.contract('q,qx,qmx,Nq->Nm', self.w_Gamma_r, self.n_r, gradbasis_test_r, gr)
        if self.hparams.get('gamma_stabilization',0)!=0:
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_l, basis_test_l, gl)
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_r, basis_test_r, gr)        
        return d

    def compute_DeepONet_coeffs(self, theta, f, etab, etat, gl, gr):
        eta = torch.zeros((etab.shape[0],2*etab.shape[1]), dtype=self.hparams['dtype'], device=self.device)
        eta[:,:etab.shape[1]] = etab
        eta[:,etab.shape[1]:] = etat
        g = torch.zeros((gl.shape[0],2*gl.shape[1]), dtype=self.hparams['dtype'], device=self.device)
        g[:,:gl.shape[1]] = gl
        g[:,gl.shape[1]:] = gr
        inputfuncs = torch.cat((theta,f,eta,g),dim=1)
        u_n = self.systemnet.forward(inputfuncs)
        return u_n
    
    def compute_VarMiON_coeffs(self, theta, f, etab, etat, gl, gr):
        systemnet = self.systemnet.forward(theta)
        eta = torch.zeros((etab.shape[0],2*etab.shape[1]), dtype=self.hparams['dtype'], device=self.device)
        eta[:,:etab.shape[1]] = etab
        eta[:,etab.shape[1]:] = etat
        g = torch.zeros((gl.shape[0],2*gl.shape[1]), dtype=self.hparams['dtype'], device=self.device)
        g[:,:gl.shape[1]] = gl
        g[:,gl.shape[1]:] = gr
        LBranch = self.LBranch_f.forward(f) + self.LBranch_eta.forward(eta) + self.LBranch_g.forward(g)
        u_n = opt_einsum.contract('nij,nj->ni', systemnet, LBranch)
        return u_n
    
    def compute_NGO_coeffs(self, F, d):
        A = self.systemnet.forward(F)
        u_n = opt_einsum.contract('nij,nj->ni', A, d)
        return u_n
    
    def compute_FEM_coeffs(self, F, d):
        K_inv = np.linalg.pinv(F)
        u_n = opt_einsum.contract('nij,nj->ni', K_inv, d)
        return u_n
    
    def compute_projection_coeffs(self, u):
        u_q = []
        for i in range(len(u)):
            u_q.append(u[i](np.array(self.xi_Omega)))
        basis_test = self.basis_test.forward(self.xi_Omega)
        basis_trial = self.basis_trial.forward(self.xi_Omega)
        u_w = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, u_q)
        M = opt_einsum.contract('q,qm,qn->mn', self.w_Omega, basis_test, basis_trial)
        M_inv = np.linalg.pinv(M)
        u_n = opt_einsum.contract('mn,Nm->Nn', M_inv, u_w)
        return u_n
    
    def init_modeltype(self):
        if self.hparams.get('modeltype',False)=='DeepONet':
            self.compute_coeffs = self.compute_DeepONet_coeffs
        if self.hparams.get('modeltype',False)=='VarMiON':
            self.compute_coeffs = self.compute_VarMiON_coeffs
        if self.hparams.get('modeltype',False)=='NGO':
            self.compute_coeffs = self.compute_NGO_coeffs
        if self.hparams.get('modeltype',False)=='FEM':
            self.compute_coeffs = self.compute_FEM_coeffs
        if self.hparams.get('modeltype',False)=='projection':
            self.compute_coeffs = self.compute_projection_coeffs
    
    def forward(self, *args):
        u_n = self.compute_coeffs(*args)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat
    
    def simforward(self, theta, f, etab, etat, gl, gr, x, u):
        theta, theta_g, f, etab, etat, gl, gr = self.discretize_input_functions(theta, f, etab, etat, gl, gr)
        theta = torch.tensor(theta, dtype=self.hparams['dtype'])
        theta_g = torch.tensor(theta_g, dtype=self.hparams['dtype'])
        f = torch.tensor(f, dtype=self.hparams['dtype'])
        etab = torch.tensor(etab, dtype=self.hparams['dtype'])
        etat = torch.tensor(etat, dtype=self.hparams['dtype'])
        gl = torch.tensor(gl, dtype=self.hparams['dtype'])
        gr = torch.tensor(gr, dtype=self.hparams['dtype'])
        if self.hparams.get('modeltype',False)=='DeepONet':
            u_n = self.compute_DeepONet_coeffs(theta, f, etab, etat, gl, gr).detach().numpy()
        if self.hparams.get('modeltype',False)=='VarMiON':
            u_n = self.compute_VarMiON_coeffs(theta, f, etab, etat, gl, gr).detach().numpy()
        if self.hparams.get('modeltype',False)=='NGO':
            F = torch.tensor(self.compute_F(theta, theta_g), dtype=self.hparams['dtype'])
            d = torch.tensor(self.compute_d(f, etab, etat, gl, gr), dtype=self.hparams['dtype'])
            u_n = self.compute_NGO_coeffs(F, d).detach().numpy()
        if self.hparams.get('modeltype',False)=='FEM':
            F = torch.tensor(self.compute_F(theta, theta_g))
            d = torch.tensor(self.compute_d(f, etab, etat, gl, gr))
            u_n = self.compute_FEM_coeffs(F, d)
        if self.hparams.get('modeltype',False)=='projection':
            u_n = self.compute_projection_coeffs(u)
        psi = self.basis_trial.forward(x)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, psi)
        return u_hat
    
    def geometry(self):
        #Quadrature
        if self.hparams['quadrature']=='uniform':
            quadrature = UniformQuadrature2D(Q=self.hparams['Q'])
        if self.hparams['quadrature']=='Gauss-Legendre':
            quadrature = GaussLegendreQuadrature2D(Q=self.hparams['Q'], n_elements=self.hparams['n_elements'])
        self.xi_Omega = quadrature.xi_Omega
        self.xi_Gamma_b = quadrature.xi_Gamma_b
        self.xi_Gamma_t = quadrature.xi_Gamma_t
        self.xi_Gamma_l = quadrature.xi_Gamma_l
        self.xi_Gamma_r = quadrature.xi_Gamma_r
        self.xi_Gamma_eta = quadrature.xi_Gamma_eta
        self.xi_Gamma_g = quadrature.xi_Gamma_g
        self.w_Omega = quadrature.w_Omega
        self.w_Gamma_b = quadrature.w_Gamma_b
        self.w_Gamma_t = quadrature.w_Gamma_t
        self.w_Gamma_l = quadrature.w_Gamma_l
        self.w_Gamma_r = quadrature.w_Gamma_r
        self.w_Gamma_eta = quadrature.w_Gamma_eta
        self.w_Gamma_g = quadrature.w_Gamma_g
        #Outward normal
        outwardnormal = UnitSquareOutwardNormal(Q=self.hparams['Q'])
        self.n_b = outwardnormal.n_b
        self.n_t = outwardnormal.n_t
        self.n_l = outwardnormal.n_l
        self.n_r = outwardnormal.n_r
        self.n_Gamma_eta = outwardnormal.n_Gamma_eta
        self.n_Gamma_g = outwardnormal.n_Gamma_g
        #Loss quadrature
        if self.hparams['quadrature_L']=='uniform':
            quadrature_L = UniformQuadrature2D(Q=self.hparams['Q_L'])
        if self.hparams['quadrature_L']=='Gauss-Legendre':
            quadrature_L = GaussLegendreQuadrature2D(Q=self.hparams['Q_L'], n_elements=self.hparams['n_elements_L'])
        self.xi_Omega_L = quadrature_L.xi_Omega
        self.w_Omega_L = quadrature_L.w_Omega
        #Basis evaluation at quadrature points
        self.psix = torch.tensor(self.basis_trial.forward(self.xi_Omega_L), dtype=self.hparams['dtype'], device=self.device)

    def configure_optimizers(self):
        optimizer = self.hparams['optimizer'](self.parameters(), lr=self.hparams['learning_rate'])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs = train_batch[:-1]
        u = train_batch[-1]
        u_hat = self.forward(*inputs)
        loss = self.hparams['loss'](self.w_Omega_L, u_hat, u)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs = val_batch[:-1]
        u = val_batch[-1]
        u_hat = self.forward(*inputs)
        loss = self.hparams['loss'](self.w_Omega_L, u_hat, u)
        metric = self.hparams['metric'](self.w_Omega_L, u_hat, u)
        self.log('val_loss', loss)
        self.log('metric', metric)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['params'] = self.params
        
    def on_fit_start(self):
        torch.set_num_threads(2)
        print('Number of threads:')
        print(torch.get_num_threads())
        self.psix = self.psix.to(self.device)
        self.w_Omega_L = torch.tensor(self.w_Omega_L).to(self.device)
        self.hparams['N_w_real'] = sum(p.numel() for p in self.systemnet.parameters())
