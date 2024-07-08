
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm.auto import tqdm, trange
from ipywidgets import IntProgress
import time
from torch import nn
from torch.nn import functional as F

from logger import *
from trainer import train
from customlosses import *
from systemnets import *
from basisfunctions import *

import sys
sys.path.insert(0, '../testproblems/darcy')
from NGO import NGO
from VarMiON import *
from DataModule import *

hparams = {'N_samples': 10000,
           'd': 2,
           'l_min': 0.5,
           'l_max': 1,
           'dtype': torch.float64,
           'precision': 64,
           'devices': [0],
           'loss': weightedrelativeL2,
           'metric': weightedrelativeL2,
           'optimizer': torch.optim.Adam, 
           'learning_rate': 1e-4,
           'batch_size': 100,
           'epochs': 20000,
           'modeltype': 'NGO',
           'data_based': False,
           'gamma_stabilization': 0,
           'quadrature': 'Gauss-Legendre',
           'Q': 160,
           'Q_L': 160,
           'h': [8,8],
           'p': 0,
           'C': -1,
           'n_elements': 8,
           'systemnet': UNet,
           'kernel_sizes': [4,4,2,2],
           'num_channels': 24,
           'NLB_outputactivation': nn.Tanhshrink(),
           'bias_NLBranch': False,
           'bias_LBranch': False,
           'scaling_equivariance': False}

hparams['N'] = np.prod(hparams['h'])
hparams['test_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'], C=hparams['C']), 
                         BSplineBasis1D(h=hparams['h'][1], p=hparams['p'], C=hparams['C'])]
hparams['trial_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'], C=hparams['C']), 
                         BSplineBasis1D(h=hparams['h'][1], p=hparams['p'], C=hparams['C'])]
# hparams['test_bases'] = [ChebyshevTBasis1D(h=hparams['h'][0]), 
#                          ChebyshevTBasis1D(h=hparams['h'][1])]
# hparams['trial_bases'] = [ChebyshevTBasis1D(h=hparams['h'][0]), 
#                          ChebyshevTBasis1D(h=hparams['h'][1])]

loaddir = '../../../trainingdata/VarMiONpaperdata/train'
logdir = '../../../nnlogs'
sublogdir = 'basisfunctions'
label = 'p0_c-1'


model = NGO
datamodule = DataModule_Darcy_MS
train(model, datamodule, hparams, loaddir, logdir, sublogdir, label)