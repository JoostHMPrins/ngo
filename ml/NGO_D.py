import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
import numpy as np
from BranchNets import *
from BSplines import *
from PODBasis import *
from Quadrature import *
from customlayers import *
from customlosses import *
import opt_einsum
import time

class NGO(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hparams.update(params['hparams'])
        #Modeltype
        self.init_modeltype()
        #Branch
        if self.hparams.get('modeltype',False)=='DeepONet':
            self.NLBranch = DeepONetBranch(params, input_dim=2*self.hparams['Q']**params['simparams']['d']+4*self.hparams['Q'], output_dim=self.hparams['h'])
        if self.hparams.get('modeltype',False)=='VarMiON':
            self.NLBranch = NLBranch_VarMiON(params)
            self.LBranch_f = LBranchNet(params, input_dim=self.hparams['Q']**params['simparams']['d'], output_dim=self.hparams['h'])
            self.LBranch_eta = LBranchNet(params, input_dim=2*self.hparams['Q'], output_dim=self.hparams['h'])
            self.LBranch_g = LBranchNet(params, input_dim=2*self.hparams['Q'], output_dim=self.hparams['h'])
        if self.hparams.get('modeltype',False)=='NGO':
            # self.NLBranch = NLBranch_NGO(params)
            self.NLBranch = UNet(params)
        #basis
        knots_POD = np.array([0,0,0,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1,1,1])
        data_dir = '../../../trainingdata/VarMiONpaperdata/test'
        x_raw = np.load(data_dir + '/x.npy')
        u_raw = np.load(data_dir + '/u.npy')
        PODBasis = BSplineInterpolatedPOD2D(x_data=x_raw[0], u_data=u_raw, h=self.hparams['h'], knots_x=knots_POD, knots_y=knots_POD, polynomial_order=3)
        BSplineBasis = BSplineBasis2D(knots_x=self.hparams['knots_1d'], knots_y=self.hparams['knots_1d'], polynomial_order=3)
        if self.hparams['test_basis']=='B-splines':
            self.basis_test = BSplineBasis
        if self.hparams['test_basis']=='POD':
            self.basis_test = PODBasis
        if self.hparams['trial_basis']=='B-splines':
            self.basis_trial = BSplineBasis
        if self.hparams['trial_basis']=='POD':
            self.basis_trial = PODBasis
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
            theta.append(theta_f[i](np.array(self.xi_Omega)))
            theta_g.append(theta_f[i](np.array(self.xi_Gamma_g)))
            f.append(f_f[i](np.array(self.xi_Omega)))
            etab.append(etab_f[i](np.array(self.xi_Gamma_b)))
            etat.append(etat_f[i](np.array(self.xi_Gamma_t)))
            gl.append(gl_f[i](np.array(self.xi_Gamma_l)))
            gr.append(gr_f[i](np.array(self.xi_Gamma_r)))
        #Convert to torch tensors
        theta = torch.tensor(np.array(theta), dtype=self.hparams['dtype'])
        theta_g = torch.tensor(np.array(theta_g), dtype=self.hparams['dtype'])
        f = torch.tensor(np.array(f), dtype=self.hparams['dtype'])
        etab = torch.tensor(np.array(etab), dtype=self.hparams['dtype'])
        etat = torch.tensor(np.array(etat), dtype=self.hparams['dtype'])
        gl = torch.tensor(np.array(gl), dtype=self.hparams['dtype'])
        gr = torch.tensor(np.array(gr), dtype=self.hparams['dtype'])
        return theta, theta_g, f, etab, etat, gl, gr
        
    def compute_K(self, theta, theta_g):
        if self.hparams['data_based']==True:
            basis_test = self.basis_test.forward(self.xi_Omega)
            K_diag = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, basis_test, theta)
            K = opt_einsum.contract('Ni,ij->Nij', K_diag, np.identity(self.hparams['h']))
        else:
            gradbasis_test = self.basis_test.grad(self.xi_Omega)
            gradbasis_trial = self.basis_trial.grad(self.xi_Omega)
            basis_test_g = self.basis_test.forward(self.xi_Gamma_g)
            gradbasis_test_g = self.basis_test.grad(self.xi_Gamma_g)
            basis_trial_g = self.basis_trial.forward(self.xi_Gamma_g)
            gradbasis_trial_g = self.basis_trial.grad(self.xi_Gamma_g)
            K = opt_einsum.contract('q,Nq,qmx,qnx->Nmn', self.w_Omega, theta, gradbasis_test, gradbasis_trial)
            K += -opt_einsum.contract('q,qm,qx,Nq,qnx->Nmn', self.w_Gamma_g, basis_test_g, self.n_Gamma_g, theta_g, gradbasis_trial_g)
            K += -opt_einsum.contract('q,qn,qx,Nq,qmx->Nmn', self.w_Gamma_g, basis_trial_g, self.n_Gamma_g, theta_g, gradbasis_test_g)
            K += self.hparams['gamma_stabilization']*opt_einsum.contract('q,qm,Nq,qn->Nmn', self.w_Gamma_g, basis_test_g, theta_g, basis_trial_g)
        return K
    
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
        u_n = self.NLBranch.forward(inputfuncs)
        return u_n
    
    def compute_VarMiON_coeffs(self, theta, f, etab, etat, gl, gr):
        NLBranch = self.NLBranch.forward(theta)
        eta = torch.zeros((etab.shape[0],2*etab.shape[1]), dtype=self.hparams['dtype'], device=self.device)
        eta[:,:etab.shape[1]] = etab
        eta[:,etab.shape[1]:] = etat
        g = torch.zeros((gl.shape[0],2*gl.shape[1]), dtype=self.hparams['dtype'], device=self.device)
        g[:,:gl.shape[1]] = gl
        g[:,gl.shape[1]:] = gr
        LBranch = self.LBranch_f.forward(f) + self.LBranch_eta.forward(eta) + self.LBranch_g.forward(g)
        u_n = opt_einsum.contract('nij,nj->ni', NLBranch, LBranch)
        return u_n
    
    def compute_NGO_coeffs(self, K, d):
        A = self.NLBranch.forward(K)
        u_n = opt_einsum.contract('nij,nj->ni', A, d)
        return u_n
    
    def compute_FEM_coeffs(self, K, d):
        K_inv = torch.linalg.inv(K)
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
        M_inv = np.linalg.inv(M)
        u_n = opt_einsum.contract('mn,Nm->Nn', M_inv, u_w)
        u_n = torch.tensor(u_n)
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
    
    # def grad(self, theta, f, etab, etat, gl, gr, K, d, psi, x, u):
    #     u_n = self.compute_coeffs(theta, f, etab, etat, gl, gr, K, d, psi, x, u)
    #     gradpsi = self.basis_trial.grad(x)
    #     gradu_hat = opt_einsum.contract('ni,noix->nox', u_n, gradpsi)
    #     return gradu_hat
    
    def simforward(self, theta, f, etab, etat, gl, gr, x, u):
        theta, theta_g, f, etab, etat, gl, gr = self.discretize_input_functions(theta, f, etab, etat, gl, gr)
        psi = torch.tensor(self.basis_trial.forward(x))
        if self.hparams['modeltype']=='DeepONet':
            u_n = self.compute_DeepONet_coeffs(theta, f, etab, etat, gl, gr)
        if self.hparams['modeltype']=='VarMiON':
            u_n = self.compute_VarMiON_coeffs(theta, f, etab, etat, gl, gr)
        if self.hparams['modeltype']=='NGO':
            K = torch.tensor(self.compute_K(theta, theta_g))
            d = torch.tensor(self.compute_d(f, etab, etat, gl, gr))
            u_n = self.compute_NGO_coeffs(K, d)
        if self.hparams['modeltype']=='FEM':
            K = torch.tensor(self.compute_K(theta, theta_g))
            d = torch.tensor(self.compute_d(f, etab, etat, gl, gr))
            u_n = self.compute_FEM_coeffs(K, d)
        if self.hparams['modeltype']=='projection':
            u_n = self.compute_projection_coeffs(u)
        u_hat = opt_einsum.contract('Nn,qn->Nq', u_n, psi).detach().numpy()
        return u_hat
    
    # def simgrad(self, theta, f, etab, etat, gl, gr, x, u):
    #     self.init_modeltype()
    #     theta, theta_g, f, etab, etat, gl, gr = self.discretize_input_functions(theta, f, etab, etat, gl, gr)
    #     K = torch.tensor(self.compute_K(theta, theta_g))
    #     d = torch.tensor(self.compute_d(f, etab, etat, gl, gr))
    #     psi = self.basis_trial.forward(x.reshape((x.shape[0]*x.shape[1],x.shape[2]))).reshape((x.shape[0],x.shape[1],self.hparams['h']))
    #     u_n = self.compute_coeffs(theta, f, etab, etat, gl, gr, K, d, psi, x, u)
    #     gradpsi = self.basis_trial.grad(x.reshape((x.shape[0]*x.shape[1],x.shape[2]))).reshape((x.shape[0],x.shape[1],self.hparams['h'],self.params['simparams']['d']))
    #     gradpsi = torch.tensor(gradpsi)
    #     gradu_hat = opt_einsum.contract('ni,noix->nox', u_n, gradpsi).detach().numpy()
    #     return gradu_hat

    def configure_optimizers(self):
        optimizer = self.hparams['optimizer'](self.parameters(), lr=self.hparams['learning_rate'])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs = train_batch[:-1]
        u = train_batch[-1]
        u_hat = self.forward(*inputs)
        loss = self.hparams['loss'](u_hat, u)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs = val_batch[:-1]
        u = val_batch[-1]
        u_hat = self.forward(*inputs)
        loss = self.hparams['loss'](u_hat, u)
        self.log('val_loss', loss)
        metric = self.hparams['metric'](u_hat, u)
        self.log('metric', metric)
        
    def geometry(self):
        #Quadrature
        if self.hparams['quadrature']=='uniform':
            quadrature = UniformQuadrature2D(Q=self.hparams['Q'])
        if self.hparams['quadrature']=='Gauss-Legendre':
            quadrature = GaussLegendreQuadrature2D(Q=self.hparams['Q'], n_elements = self.basis_test.num_basis_1d - self.basis_test.p)
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
        if self.hparams['quadrature']=='uniform':
            quadrature_L = UniformQuadrature2D(Q=self.hparams['Q_L'])
        if self.hparams['quadrature']=='Gauss-Legendre':
            quadrature_L = GaussLegendreQuadrature2D(Q=self.hparams['Q_L'], n_elements = self.basis_test.num_basis_1d - self.basis_test.p)
        self.xi_Omega_L = quadrature_L.xi_Omega
        self.psix = torch.tensor(self.basis_trial.forward(self.xi_Omega_L), dtype=self.hparams['dtype'], device=self.device)

    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint['params'] = self.params
        
    def on_fit_start(self):
        torch.set_num_threads(2)
        print('Number of threads:')
        print(torch.get_num_threads())
        print(self.device)
        self.psix = self.psix.to(self.device)