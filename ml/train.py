import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm.auto import tqdm, trange
from ipywidgets import IntProgress
import time
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional.regression import mean_absolute_percentage_error

import sys
sys.path.insert(0, '../gitlab/ngo-pde-gk/ml')
from NGO_D import *
from VarMiON import *
from DataModule import *
from logger import *
from trainer import train
from customlosses import *

hparams = {'devices': [1],
           'dtype': torch.float32,
           'precision': 32,
           'loss': F.mse_loss,
           'metric': L2_scaled,
           'optimizer': torch.optim.Adam, 
           'learning_rate': 1e-4,
           'batch_size': 100,
           'epochs': 10000,
           'modeltype': 'NGO',
           'gamma_stabilization': 0,
           'data_based': False,
           'test_basis': 'B-splines',
           'trial_basis': 'B-splines',
           'quadrature': 'Gauss-Legendre',
           'Q': 100,
           'Q_L': 100,
           'h': 64,
           # 'knots_1d': np.array([0,0,0,0, 1/9, 2/9, 3/9, 4/9, 5/9, 6/9, 7/9, 8/9, 1,1,1,1]),
           'knots_1d': np.array([0,0,0,0, 0.2, 0.4, 0.6, 0.8, 1,1,1,1]),
           'UNet': True,
           'num_channels': 24,
           'kernel_sizes': [4,4,2,2],
           'bias_NLBranch': False,
           'bias_LBranch': False,
           'NLB_outputactivation': nn.Tanhshrink(),
           'scale_invariance': False,
           'symgroupavg': False}

loaddir = '../../../trainingdata/VarMiONpaperdata/train'
logdir = '../../../nnlogs'
sublogdir = 'test'
label = 'new'

model = NGO
# model = RegularNN
# datamodule = DataModule_hc2d
datamodule = DataModule_Darcy_MS

train(model, datamodule, hparams, loaddir, logdir, sublogdir, label)