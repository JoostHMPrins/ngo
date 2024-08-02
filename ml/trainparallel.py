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

import argparse
parser = argparse.ArgumentParser(description='this program does this and that')
parser.add_argument('--arg', type=str, required=True, help='training label id')
args = parser.parse_args()

hparams = {}

#Training data
hparams['N_samples'] = 10000
hparams['d'] = 2
hparams['l_min'] = float(args.arg)
hparams['l_max'] = float(args.arg)

#Training settings
hparams['dtype'] = torch.float32
hparams['precision'] = 32
hparams['devices'] = [1]
hparams['loss'] = weightedrelativeL2
hparams['metric'] = weightedrelativeL2
hparams['optimizer'] = torch.optim.Adam 
hparams['learning_rate'] = 1e-3
hparams['batch_size'] = 100
hparams['epochs'] = 5000

#System net
hparams['modeltype'] = 'NGO'
hparams['model/data'] = 'model'
hparams['gamma_stabilization'] = 0
hparams['systemnet'] = UNet
hparams['kernel_sizes'] = [5,5,2,2,2,2,5,5]
hparams['N_w'] = 30000
hparams['bottleneck_size'] = 20
hparams['NLB_outputactivation'] = nn.Tanhshrink()
hparams['bias_NLBranch'] = False
hparams['bias_LBranch'] = False

#Bases
hparams['h'] = [10,10]
hparams['p'] = 3
hparams['C'] = 2
hparams['N'] = np.prod(hparams['h'])
hparams['test_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'], C=hparams['C']), 
                         BSplineBasis1D(h=hparams['h'][1], p=hparams['p'], C=hparams['C'])]
hparams['trial_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'], C=hparams['C']), 
                         BSplineBasis1D(h=hparams['h'][1], p=hparams['p'], C=hparams['C'])]

#Quadrature
hparams['quadrature'] = 'Gauss-Legendre'
hparams['n_elements'] = max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q'] = 33*hparams['n_elements']

hparams['quadrature_L'] = 'Gauss-Legendre'
hparams['n_elements_L'] = max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q_L'] = 33*hparams['n_elements_L']

#Symmetries
hparams['scaling_equivariance'] = False

loaddir = None
logdir = '../../../nnlogs'
sublogdir = 'lambda_train'
label = str(args.arg)


model = NGO
datamodule = DataModule_Darcy_MS
train(model, datamodule, hparams, loaddir, logdir, sublogdir, label)