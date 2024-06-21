import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
import numpy as np
from BranchNets import *
from BSplines import *
from PODBasis import *
from customlayers import *
from customlosses import *
from Quadrature import *
import opt_einsum

class NGO(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hparams.update(params['hparams'])
        #Modeltype
        self.init_forward()
        #Branch
        if self.hparams.get('modeltype',False)=='DeepONet':
            self.NLBranch = DeepONetBranch(params, input_dim=2*self.hparams['Q']**params['simparams']['d']+2*self.hparams['Q'], output_dim=self.hparams['h'])
        if self.hparams.get('modeltype',False)=='VarMiON':
            self.NLBranch = NLBranch_VarMiON(params)
            self.LBranch_f = LBranchNet(params, input_dim=self.hparams['Q']**params['simparams']['d'], output_dim=self.hparams['h'])
            self.LBranch_eta = LBranchNet(params, input_dim=2*self.hparams['Q'], output_dim=self.hparams['h'])
        if self.hparams.get('modeltype',False)=='NGO':
            self.NLBranch = UNet(params)
            # self.NLBranch = NLBranch_NGO(params)
        #Trunk
        knots_BS = self.hparams['knots_1d']
        knots_POD = np.array([0,0,0,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1,1,1])
        data_dir = '../../../trainingdata/VarMiONpaperdata/test'
        x_raw = np.load(data_dir + '/x.npy')
        u_raw = np.load(data_dir + '/u.npy')
        PODBasis = BSplineInterpolatedPOD2D(x_data=x_raw[0], u_data=u_raw, h=self.hparams['h'], knots_x=knots_POD, knots_y=knots_POD, polynomial_order=3)
        BSplineBasis = BSplineBasis2D(knots_x=knots_BS, knots_y=knots_BS, polynomial_order=3)
        if self.hparams['test_basis']=='B-splines':
            self.Trunk_test = BSplineBasis
        if self.hparams['test_basis']=='POD':
            self.Trunk_test = PODBasis
        if self.hparams['trial_basis']=='B-splines':
            self.Trunk_trial = BSplineBasis
        if self.hparams['trial_basis']=='POD':
            self.Trunk_trial = PODBasis
        #Geometry and quadrature
        self.geometry()
        self = self.to(self.hparams['dtype'])
        
    def discretize_input_functions(self, theta_f, f_f, etab_f, etat_f):
        theta = []
        theta_g = []
        f = []
        etab = []
        etat = []
        for i in range(len(theta_f)):
            theta.append(theta_f[i](np.array(self.xi_Omega)))
            theta_g.append(theta_f[i](np.array(self.xi_Gamma_g)))
            f.append(f_f[i](np.array(self.xi_Omega)))
            etab.append(etab_f[i](np.array(self.xi_Gamma_b)))
            etat.append(etat_f[i](np.array(self.xi_Gamma_t)))
        #Convert to torch tensors
        theta = torch.tensor(np.array(theta), dtype=self.hparams['dtype'])
        theta_g = torch.tensor(np.array(theta_g), dtype=self.hparams['dtype'])
        f = torch.tensor(np.array(f), dtype=self.hparams['dtype'])
        etab = torch.tensor(np.array(etab), dtype=self.hparams['dtype'])
        etat = torch.tensor(np.array(etat), dtype=self.hparams['dtype'])
        return theta, theta_g, f, etab, etat
        
    def compute_K(self, theta, theta_g):
        gradTrunk_test = self.Trunk_test.grad(self.xi_Omega)
        gradTrunk_trial = self.Trunk_trial.grad(self.xi_Omega)
        Trunk_test_g = self.Trunk_test.forward(self.xi_Gamma_g)
        gradTrunk_test_g = self.Trunk_test.grad(self.xi_Gamma_g)
        Trunk_trial_g = self.Trunk_trial.forward(self.xi_Gamma_g)
        gradTrunk_trial_g = self.Trunk_trial.grad(self.xi_Gamma_g)
        K = opt_einsum.contract('q,Nq,qmx,qnx->Nmn', self.w_Omega, theta, gradTrunk_test, gradTrunk_trial)
        K += -opt_einsum.contract('q,qm,qx,Nq,qnx->Nmn', self.w_Gamma_g, Trunk_test_g, self.n_Gamma_g, theta_g, gradTrunk_trial_g)
        K += -opt_einsum.contract('q,qn,qx,Nq,qmx->Nmn', self.w_Gamma_g, Trunk_trial_g, self.n_Gamma_g, theta_g, gradTrunk_test_g)
        return K
    
    def compute_d(self, f, etab, etat):
        Trunk_test = self.Trunk_test.forward(self.xi_Omega)
        Trunk_test_b = self.Trunk_test.forward(self.xi_Gamma_b)
        Trunk_test_t = self.Trunk_test.forward(self.xi_Gamma_t)
        d = opt_einsum.contract('q,qm,Nq->Nm', self.w_Omega, Trunk_test, f)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_b, Trunk_test_b, etab)
        d += opt_einsum.contract('q,qm,Nq->Nm', self.w_Gamma_t, Trunk_test_t, etat)
        return d

    def forward_DeepONet(self, theta, f, etab, etat, K, d, psi, x):
        eta = torch.zeros((etab.shape[0],2*etab.shape[1]), dtype=self.hparams['dtype'], device=self.device)
        eta[:,:etab.shape[1]] = etab
        eta[:,etab.shape[1]:] = etat
        inputfuncs = torch.cat((theta,f,eta),dim=1)
        u_n = self.NLBranch.forward(inputfuncs)
        u_hat = opt_einsum.contract('ni,noi->no', u_n, psi)
        return u_hat
    
    def forward_VarMiON(self, theta, f, etab, etat, K, d, psi, x):
        NLBranch = self.NLBranch.forward(theta)
        eta = torch.zeros((etab.shape[0],2*etab.shape[1]), dtype=self.hparams['dtype'], device=self.device)
        eta[:,:etab.shape[1]] = etab
        eta[:,etab.shape[1]:] = etat
        LBranch = self.LBranch_f.forward(f) + self.LBranch_eta.forward(eta)
        u_n = opt_einsum.contract('nij,nj->ni', NLBranch, LBranch)
        u_hat = opt_einsum.contract('ni,noi->no', u_n, psi)
        return u_hat
    
    def forward_NGO(self, theta, f, etab, etat, K, d, psi, x):
        K_inv = self.NLBranch.forward(K)
        u_n = opt_einsum.contract('nij,nj->ni', K_inv, d)
        u_hat = opt_einsum.contract('ni,noi->no', u_n, psi)
        return u_hat
    
    def forward_FEM(self, theta, f, etab, etat, K, d, psi, x):
        K_inv = torch.linalg.inv(K)
        u_n = opt_einsum.contract('nij,nj->ni', K_inv, d)
        u_hat = opt_einsum.contract('ni,noi->no', u_n, psi)
        return u_hat
    
    def init_forward(self):
        if self.hparams.get('modeltype',False)=='DeepONet':
            self.forwardfunction = self.forward_DeepONet
        if self.hparams.get('modeltype',False)=='VarMiON':
            self.forwardfunction = self.forward_VarMiON
        if self.hparams.get('modeltype',False)=='NGO':
            self.forwardfunction = self.forward_NGO
    
    def forward(self, theta, f, etab, etat, K, d, psi, x):
        u_hat = self.forwardfunction(theta, f, etab, etat, K, d, psi, x) 
        return u_hat
    
    def simforward(self, theta, f, etab, etat, x):
        theta, theta_g, f, etab, etat = self.discretize_input_functions(theta, f, etab, etat)
        K = torch.tensor(self.compute_K(theta, theta_g))
        d = torch.tensor(self.compute_d(f, etab, etat))
        psi = torch.tensor(self.Trunk_trial.forward(x.reshape((x.shape[0]*x.shape[1],x.shape[2]))).reshape((x.shape[0],x.shape[1],self.hparams['h'])))
        u_hat = self.forward(theta, f, etab, etat, K, d, psi, x).detach().numpy()
        return u_hat
    
    def simforward_FEM(self, theta, f, etab, etat, x):
        theta, theta_g, f, etab, etat = self.discretize_input_functions(theta, f, etab, etat)
        K = torch.tensor(self.compute_K(theta, theta_g))
        d = torch.tensor(self.compute_d(f, etab, etat))
        psi = torch.tensor(self.Trunk_trial.forward(x.reshape((x.shape[0]*x.shape[1],x.shape[2]))).reshape((x.shape[0],x.shape[1],self.hparams['h'])))
        u_hat = self.forward_FEM(theta, f, etab, etat, K, d, psi, x).detach().numpy()
        return u_hat

    def configure_optimizers(self):
        optimizer = self.hparams['optimizer'](self.parameters(), lr=self.hparams['learning_rate'])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        theta, f, etab, etat, K, d, psi, x, u = train_batch
        u_hat = self.forward(theta, f, etab, etat, K, d, psi, x)
        loss = self.hparams['loss'](u_hat, u)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        theta, f, etab, etat, K, d, psi, x, u = val_batch
        u_hat = self.forward(theta, f, etab, etat, K, d, psi, x)
        loss = self.hparams['loss'](u_hat, u)
        self.log('val_loss', loss)
        metric = self.hparams['metric'](u_hat, u)
        self.log('metric', metric)
        
    def geometry(self):
        #Quadrature
        if self.hparams['quadrature']=='uniform':
            quadrature = UniformQuadrature2D(Q=self.hparams['Q'])
        if self.hparams['quadrature']=='Gauss-Legendre':
            quadrature = GaussLegendreQuadrature2D(Q=self.hparams['Q'], n_elements = self.Trunk_test.num_basis_1d - self.Trunk_test.p)
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
            
    def on_save_checkpoint(self, checkpoint):
        checkpoint['params'] = self.params