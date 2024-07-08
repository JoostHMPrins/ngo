import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from customlayers import GaussianRBF, expand_D8
from customlosses import *
    
class NLBranchNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        self.layers.append(nn.ConvTranspose2d(in_channels=2, out_channels=16, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch',True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=16))
        self.layers.append(nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch',True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=32))
        self.layers.append(nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch',True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch',True)))
        if self.hparams['NLB_outputactivation']!=None:
            self.layers.append(self.hparams['NLB_outputactivation'])
        self = self.to(self.hparams['dtype'])

    def forward(self, x):
        x = x.reshape((x.shape[0],x.shape[3],x.shape[1],x.shape[2])).double()
        for layer in self.layers:
            x = layer(x)
        y = x.squeeze()
        return y
    
    
class LBranchNet(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, output_dim, bias=self.hparams.get('bias_LBranch',True)))
        self = self.to(self.hparams['dtype'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        y = x
        return y
    
    
class BMVarMiON(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hparams.update(params['hparams'])
        self.NLBranch = NLBranchNet(params)
        # self.LBranchx = LBranchNet(params, input_dim=44, output_dim=72)
        # self.LBranchy = LBranchNet(params, input_dim=44, output_dim=72)
        self.LBranch = LBranchNet(params, input_dim=44, output_dim=72)
        self.Trunk = GaussianRBF(params, input_dim=params['simparams']['d'], output_dim=72)
        self = self.to(self.hparams['dtype'])
        
    def translate(self, x_in, x_D):
        x_cm = torch.mean(x_D, axis=1)
        x_in_t = x_in - x_cm[:,None,None,:]
        x_D_t = x_D - x_cm[:,None,:]
        return x_cm, x_in_t, x_D_t
    
    def translateback(self, x_cm, x_out_t):
        x_out = x_out_t + x_cm[:,None,None,:]
        return x_out

    def scale(self, x_in, x_D):
        r = torch.sqrt(torch.sum(x_D**2, axis=-1))
        r_max = torch.amax(r, axis=-1)
        x_in_s = x_in/r_max[:,None,None,None]
        x_D_s = x_D/r_max[:,None,None]
        return r_max, x_in_s, x_D_s
    
    def scaleback(self, r_max, x_out_s):
        x_out = x_out_s*r_max[:,None,None,None]
        return x_out
    
    def rotate_flip(self, x_in, x_D):
        #Calculation of angle of point with maximum radius
        r_D = torch.sqrt(torch.sum(x_D**2, axis=-1))
        i_max = torch.argmax(r_D, axis=-1)
        x0_D_max = x_D[torch.arange(x_D.shape[0]), i_max, 0]
        x1_D_max = x_D[torch.arange(x_D.shape[0]), i_max, 1]
        theta_max = torch.arctan(x1_D_max/x0_D_max)
        #Rotation matrix
        R00 = torch.cos(-theta_max)
        R01 = -torch.sin(-theta_max)
        R10 = torch.sin(-theta_max)
        R11 = torch.cos(-theta_max)
        R = torch.zeros(theta_max.shape[0],2,2).double().to(self.device)
        R[:,0,0] = R00
        R[:,0,1] = R01
        R[:,1,0] = R10
        R[:,1,1] = R11
        #Rotate point with maximum radius such that it aligns with x-axis
        x_D_r = torch.einsum('nij,npj->npi', R, x_D)
        x_in_r = torch.einsum('nij,nxyj->nxyi', R, x_in)
        #If necessary, rotate 180 degrees to align with positive x-axis
        i_R180 = -torch.amin(x_D_r[:,:,0], axis=-1) > torch.amax(x_D_r[:,:,0], axis=-1)
        x_D_r[i_R180] = -x_D_r[i_R180]
        x_in_r[i_R180] = -x_in_r[i_R180]
        #If point with highest |y| lies below y=0, mirror all points in x-axis
        i_flip = -torch.amin(x_D_r[:,:,1], axis=-1) > torch.amax(x_D_r[:,:,1], axis=-1)
        x_D_rf = torch.clone(x_D_r)
        x_in_rf = torch.clone(x_in_r)
        x_D_rf[i_flip,:,1] = -x_D_rf[i_flip,:,1]
        x_in_rf[i_flip,:,:,1] = -x_in_rf[i_flip,:,:,1]
        return theta_max, i_R180, i_flip, x_in_rf, x_D_rf
    
    def rotate_flip_back(self, theta_max, i_R180, i_flip, x_out_rf):
        #Rotation matrix
        R00 = torch.cos(theta_max)
        R01 = -torch.sin(theta_max)
        R10 = torch.sin(theta_max)
        R11 = torch.cos(theta_max)
        R = torch.zeros(theta_max.shape[0],2,2).double().to(self.device)
        R[:,0,0] = R00
        R[:,0,1] = R01
        R[:,1,0] = R10
        R[:,1,1] = R11
        #Flip back
        x_out_r = torch.clone(x_out_rf)
        x_out_r[i_flip,:,:,1] = -x_out_rf[i_flip,:,:,1]
        #Rotate back 180 degrees
        x_out_r[i_R180] = -x_out_r[i_R180]
        #Rotate back
        x_out = torch.einsum('nij,nxyj->nxyi', R.double(), x_out_r.double())      
        return x_out
        
    def forward(self, x_in, x_D, xi):
        if self.hparams.get('translation_invariance',False)==True:
            x_cm, x_in, x_D = self.translate(x_in, x_D)
        if self.hparams.get('scale_invariance',False)==True:
            r_max, x_in, x_D = self.scale(x_in, x_D)
        if self.hparams.get('rotref_invariance',False)==True:
            theta_max, i_R180, i_flip, x_in, x_D = self.rotate_flip(x_in, x_D)
        NLBranch = self.NLBranch.forward(x_in)
        # NLBranchx = NLBranch[:,0,:,:]
        # NLBranchy = NLBranch[:,1,:,:]
        LBranchx = self.LBranch.forward(x_D[:,:,0])
        LBranchy = self.LBranch.forward(x_D[:,:,1])
        Branchx = torch.einsum('nij,nj->ni', NLBranch, LBranchx)
        Branchy = torch.einsum('nij,nj->ni', NLBranch, LBranchy)
        Trunk = self.Trunk.forward(xi)
        x_out_hat_x = torch.einsum('ni,noi->no', Branchx, Trunk).reshape(x_in.shape[0],x_in.shape[1],x_in.shape[2])
        x_out_hat_y = torch.einsum('ni,noi->no', Branchy, Trunk).reshape(x_in.shape[0],x_in.shape[1],x_in.shape[2])
        x_out_hat = torch.zeros(x_in.shape, device=self.device)
        x_out_hat[:,:,:,0] = x_out_hat_x
        x_out_hat[:,:,:,1] = x_out_hat_y
        if self.hparams.get('skipconnection',False)==True:
            x_out_hat = x_in + x_out_hat
        if self.hparams.get('rotref_invariance',False)==True:
            x_out_hat = self.rotate_flip_back(theta_max, i_R180, i_flip, x_out_hat)
        if self.hparams.get('scale_invariance',False)==True:
            x_out_hat = self.scaleback(r_max, x_out_hat)
        if self.hparams.get('translation_invariance',False)==True:
            x_out_hat = self.translateback(x_cm, x_out_hat)
        return x_out_hat    

    def configure_optimizers(self):
        optimizer = self.hparams['optimizer'](self.parameters(), lr=self.hparams['learning_rate'])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_in, x_D, xi, x_out = train_batch
        x_out_hat = self.forward(x_in, x_D, xi)
        loss = 0
        for i in range(len(self.hparams['loss_coeffs'])):
            loss = loss + self.hparams['loss_coeffs'][i]*self.hparams['loss_terms'][i](x_out_hat, x_out)
        loss = loss/sum(self.hparams['loss_coeffs'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x_in, x_D, xi, x_out = val_batch
        x_out_hat = self.forward(x_in, x_D, xi)
        loss = 0
        for i in range(len(self.hparams['loss_coeffs'])):
            loss = loss + self.hparams['loss_coeffs'][i]*self.hparams['loss_terms'][i](x_out_hat, x_out)
        loss = loss/sum(self.hparams['loss_coeffs'])
        self.log('val_loss', loss)
        metric = self.hparams['metric'](x_out_hat, x_out)
        self.log('metric', metric)
        
    def on_before_zero_grad(self, optimizer):
        if self.hparams.get('bound_mus',False)==True:
            for name, p in self.Trunk.named_parameters():
                if name=='mus':
                    p.data.clamp_(0, 1.0)
            
    def on_save_checkpoint(self, checkpoint):
        checkpoint['params'] = self.params