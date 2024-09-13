import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
import numpy as np
import opt_einsum

import sys
sys.path.insert(0, '../../ml')
from systemnets import *
from basisfunctions import *
from quadrature import *
from customlayers import *
from customlosses import *

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
        #Bases
        self.basis_test = TensorizedBasis(self.hparams['test_bases'])
        self.basis_trial = TensorizedBasis(self.hparams['trial_bases'])
        self.basis_test_F = TensorizedBasis(self.hparams['test_bases_F'])
        self.basis_trial_F = TensorizedBasis(self.hparams['trial_bases_F'])
        #Geometry and quadrature
        self.geometry()
        self = self.to(self.hparams['dtype'])

    def discretize_input_functions(self, theta, f, etab, etat, gl, gr):
        theta_d = discretize_functions(theta, self.xi_Omega, device=self.used_device)
        theta_g_d = discretize_functions(theta, self.xi_Gamma_g, device=self.used_device)
        f_d = discretize_functions(f, self.xi_Omega, device=self.used_device)
        etab_d = discretize_functions(etab, self.xi_Gamma_b, device=self.used_device)
        etat_d = discretize_functions(etat, self.xi_Gamma_t, device=self.used_device)
        gl_d = discretize_functions(gl, self.xi_Gamma_l, device=self.used_device)
        gr_d = discretize_functions(gr, self.xi_Gamma_r, device=self.used_device)
        return theta_d, theta_g_d, f_d, etab_d, etat_d, gl_d, gr_d
    
    def discretize_output_function(self, u):
        u_d = discretize_functions(u, self.xi_Omega_L, device=self.used_device)
        return u_d

    def compute_F(self, theta, theta_g):
        if  self.hparams['modeltype']=='data NGO':
            basis_test_F = self.basis_test_F.forward(self.xi_Omega.cpu().numpy())
            F = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega.cpu().numpy(), basis_test_F, theta.cpu().numpy()).reshape((theta.shape[0],self.hparams['h'][0],self.hparams['h'][1]))
        if  self.hparams['modeltype']=='matrix data NGO':
            basis_test_F = self.basis_test_F.forward(self.xi_Omega.cpu().numpy())
            basis_trial_F = self.basis_trial_F.forward(self.xi_Omega.cpu().numpy())
            F = opt_einsum.contract('q,Nq,qm,qn->Nmn', self.w_Omega.cpu().numpy(), theta.cpu().numpy(), basis_test_F, basis_trial_F)
        if  self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='FEM':
            gradbasis_test_F = self.basis_test_F.grad(self.xi_Omega.cpu().numpy())
            gradbasis_trial_F = self.basis_trial_F.grad(self.xi_Omega.cpu().numpy())
            basis_test_g_F = self.basis_test_F.forward(self.xi_Gamma_g.cpu().numpy())
            gradbasis_test_g_F = self.basis_test_F.grad(self.xi_Gamma_g.cpu().numpy())
            basis_trial_g_F = self.basis_trial_F.forward(self.xi_Gamma_g.cpu().numpy())
            gradbasis_trial_g_F = self.basis_trial_F.grad(self.xi_Gamma_g.cpu().numpy())
            F = opt_einsum.contract('q,Nq,qmx,qnx->Nmn', self.w_Omega.cpu().numpy(), theta.cpu().numpy(), gradbasis_test_F, gradbasis_trial_F)
            F += -opt_einsum.contract('q,qm,qx,Nq,qnx->Nmn', self.w_Gamma_g.cpu().numpy(), basis_test_g_F, self.n_Gamma_g.cpu().numpy(), theta_g.cpu().numpy(), gradbasis_trial_g_F)
            F += -opt_einsum.contract('q,qn,qx,Nq,qmx->Nmn', self.w_Gamma_g.cpu().numpy(), basis_trial_g_F, self.n_Gamma_g.cpu().numpy(), theta_g.cpu().numpy(), gradbasis_test_g_F)
        if self.hparams.get('gamma_stabilization',0)!=0:
            F += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_g.cpu().numpy(), basis_test_g_F, theta_g.cpu().numpy(), basis_trial_g_F)
        return F
    
    def compute_d(self, f, etab, etat, gl, gr):
        basis_test = self.basis_test.forward(self.xi_Omega.cpu().numpy())
        basis_test_b = self.basis_test.forward(self.xi_Gamma_b.cpu().numpy())
        basis_test_t = self.basis_test.forward(self.xi_Gamma_t.cpu().numpy())
        basis_test_l = self.basis_test.forward(self.xi_Gamma_l.cpu().numpy())
        basis_test_r = self.basis_test.forward(self.xi_Gamma_r.cpu().numpy())
        gradbasis_test_l = self.basis_test.grad(self.xi_Gamma_l.cpu().numpy())
        gradbasis_test_r = self.basis_test.grad(self.xi_Gamma_r.cpu().numpy())
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega.cpu().numpy(), basis_test, f.cpu().numpy())
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_b.cpu().numpy(), basis_test_b, etab.cpu().numpy())
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_t.cpu().numpy(), basis_test_t, etat.cpu().numpy())
        d -= opt_einsum.contract('q,qx,qmx,Nq->Nm', self.w_Gamma_l.cpu().numpy(), self.n_l.cpu().numpy(), gradbasis_test_l, gl.cpu().numpy())
        d -= opt_einsum.contract('q,qx,qmx,Nq->Nm', self.w_Gamma_r.cpu().numpy(), self.n_r.cpu().numpy(), gradbasis_test_r, gr.cpu().numpy())
        if self.hparams.get('gamma_stabilization',0)!=0:
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_l.cpu().numpy(), basis_test_l, gl.cpu().numpy())
            d += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_r.cpu().numpy(), basis_test_r, gr.cpu().numpy())        
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
        inputfuncs = torch.stack((theta,f,eta,g),dim=1)
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
        inputfuncs = torch.stack((theta,f,eta,g),dim=1)
        A = self.systemnet.forward(inputfuncs)
        u_n = torch.sum(A, dim=-2)
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
    
    def FNO_forward(self, theta, f, etab, etat, gl, gr):
        theta = theta.reshape((theta.shape[0],self.hparams['Q'],self.hparams['Q']))
        f = f.reshape((f.shape[0],self.hparams['Q'],self.hparams['Q']))
        eta = torch.zeros((etab.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        eta[:,:,0] = etab
        eta[:,:,-1] = etat
        g = torch.zeros((gl.shape[0],self.hparams['Q'],self.hparams['Q']), dtype=self.hparams['dtype'], device=self.used_device)
        g[:,0,:] = gl
        g[:,-1,:] = gr
        inputfuncs = torch.stack((theta,f,eta,g),dim=1)
        u_hat = self.systemnet.forward(inputfuncs).reshape((theta.shape[0],self.hparams['Q_L']**self.hparams['d']))
        return u_hat
    
    def NGO_forward(self, F, d):
        A = self.systemnet.forward(F)
        u_n = opt_einsum.contract('nij,nj->ni', A, d)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat
    
    def CH_forward(self, F, d):
        c = self.systemnet.forward(F)
        Identity = torch.eye(self.hparams['N'], dtype=self.hparams['dtype'], device=self.used_device)
        F_i = torch.tile(Identity, (F.shape[0],1,1))
        A_i = c[:,1,None,None]*F_i
        # A_i = A_i/torch.amax(torch.abs(A_i), dim=(-1,-2))[:,None,None]
        u_n = opt_einsum.contract('nij,nj->ni', A_i, d)
        for i in range(2, self.hparams['k']):
            F_i = torch.matmul(F, F_i)
            # F_i = F_i/torch.amax(torch.abs(F_i), dim=(-1,-2))[:,None,None]
            A_i = c[:,i,None,None]*F_i
            # A_i = A_i/torch.amax(torch.abs(A_i), dim=(-1,-2))[:,None,None]
            u_n = u_n + opt_einsum.contract('nij,nj->ni', A_i, d)
        u_n = -1/c[:,0,None]*u_n
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, self.psix)
        return u_hat    

    def FEM_forward(self, F, d):
        K_inv = torch.linalg.pinv(F)
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
        if self.hparams['modeltype']=='CH':
            self.forwardfunction = self.CH_forward
        if self.hparams['modeltype']=='FEM':
            self.forwardfunction = self.FEM_forward
        if self.hparams['modeltype']=='projection':
            self.forwardfunction = self.projection_forward
    
    def forward(self, *args):
        u_hat = self.forwardfunction(*args)
        return u_hat
    
    def simforward(self, theta, f, etab, etat, gl, gr, x, u):
        self.geometry()
        theta, theta_g, f, etab, etat, gl, gr = self.discretize_input_functions(theta, f, etab, etat, gl, gr)
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO' or self.hparams['modeltype']=='FEM':
            F = torch.tensor(self.compute_F(theta, theta_g), dtype=self.hparams['dtype'], device=self.used_device)
            d = torch.tensor(self.compute_d(f, etab, etat, gl, gr), dtype=self.hparams['dtype'], device=self.used_device)
        theta = torch.tensor(theta, dtype=self.hparams['dtype'], device=self.used_device)
        f = torch.tensor(f, dtype=self.hparams['dtype'], device=self.used_device)
        etab = torch.tensor(etab, dtype=self.hparams['dtype'], device=self.used_device)
        etat = torch.tensor(etat, dtype=self.hparams['dtype'], device=self.used_device)
        gl = torch.tensor(gl, dtype=self.hparams['dtype'], device=self.used_device)
        gr = torch.tensor(gr, dtype=self.hparams['dtype'], device=self.used_device)    
        self.psix = torch.tensor(self.basis_trial.forward(x.cpu()), dtype=self.hparams['dtype'], device=self.used_device)
        if self.hparams['modeltype']=='NN':
            u_hat = self.NN_forward(theta, f, etab, etat, gl, gr)
        if self.hparams['modeltype']=='DeepONet':
            u_hat = self.DeepONet_forward(theta, f, etab, etat, gl, gr)
        if self.hparams['modeltype']=='VarMiON':
            u_hat = self.VarMiON_forward(theta, f, etab, etat, gl, gr)
        if self.hparams['modeltype']=='FNO':
            u_hat = self.FNO_forward(theta, f, etab, etat, gl, gr)
        if self.hparams['modeltype']=='model NGO' or self.hparams['modeltype']=='data NGO':
            u_hat = self.NGO_forward(F, d)
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
        #Basis evaluation at quadrature points
        self.psix = torch.tensor(self.basis_trial.forward(self.xi_Omega_L.cpu().numpy()), dtype=self.hparams['dtype'], device=self.used_device)

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
                F = inputs[0]
                d = inputs[1]
                A_hat = self.systemnet.forward(F)
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
            F = inputs[0]
            d = inputs[1]
            A_hat = self.systemnet.forward(F)
            loss += self.hparams['matrix_loss'](torch.matmul(F, torch.matmul(A_hat,F)), F)
            # loss += self.hparams['matrix_loss'](torch.matmul(A_hat, torch.matmul(F,A_hat)), A_hat)
            # u_hat_1 = opt_einsum.contract('nij,njk,nkl,nl->ni', A_hat,F,A_hat,d)
            # u_hat_2 = opt_einsum.contract('nij,nj->ni', A_hat,d)
            # loss += self.hparams['matrix_loss'](u_hat_1,u_hat_2)
        metric = self.hparams['metric'](self.w_Omega_L, u_hat, u)
        self.metric.append(metric)
        print(loss)
        self.log('val_loss', loss)
        self.log('metric', metric)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['hparams'] = self.hparams
        
    def on_fit_start(self):
        self.used_device = self.device
        torch.set_num_threads(2)
        self.psix = self.psix.to(self.used_device)
        self.w_Omega_L = torch.tensor(self.w_Omega_L).to(self.used_device)
        self.systemnet.device = self.used_device

    def on_validation_epoch_end(self):
        if self.hparams['switch_threshold']!=None:
            if self.metric[-1]<self.hparams['switch_threshold']:
                    self.optimizer_idx = 1
                    self.hparams['batch_size'] = self.hparams['N_samples']