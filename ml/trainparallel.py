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
from NeuralOperator import NeuralOperator
from VarMiON import *
from DataModule import *

import argparse
parser = argparse.ArgumentParser(description='this program does this and that')
parser.add_argument('--arg', type=str, required=True, help='training label id')
args = parser.parse_args()

hparams = {}

#Test problem
import sys
sys.path.insert(0, '../testproblems/darcy')
from NGO import NGO
from DataModule import *

#Training data
hparams['N_samples'] = 10000
hparams['d'] = 2
hparams['l_min'] = 0.5
hparams['l_max'] = 1

#Training settings
hparams['dtype'] = torch.float64
hparams['precision'] = 64
hparams['devices'] = [1]
hparams['solution_loss'] = weightedrelativeL2
hparams['matrix_loss'] = None #relativevectorfrobeniusnorm
hparams['metric'] = weightedrelativeL2
hparams['optimizer1'] = torch.optim.Adam
hparams['optimizer2'] = torch.optim.LBFGS
hparams['switch_threshold'] = None #5e-3
hparams['learning_rate'] = 1e-3
hparams['batch_size'] = 100
hparams['epochs'] = 5000

#System net'
hparams['in_channels'] = 1
hparams['modeltype'] = 'NGO'
hparams['model/data'] = 'model'
hparams['gamma_stabilization'] = 0
hparams['systemnet'] = InvCNN
hparams['kernel_sizes'] = [1,1,1,int(args.arg),int(args.arg),1,1,1]
hparams['N_w'] = 30000
hparams['bottleneck_size'] = 1
hparams['NLB_outputactivation'] = None#nn.Tanhshrink()
hparams['bias_NLBranch'] = False
hparams['bias_LBranch'] = False

#Bases
hparams['h'] = [10,10]
hparams['h_F'] = [10,10]
hparams['p'] = 3
hparams['C'] = 2
hparams['N'] = np.prod(hparams['h'])
hparams['N_F'] = np.prod(hparams['h_F'])
hparams['test_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'], C=hparams['C']), 
                         BSplineBasis1D(h=hparams['h'][1], p=hparams['p'], C=hparams['C'])]
hparams['trial_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'], C=hparams['C']), 
                         BSplineBasis1D(h=hparams['h'][1], p=hparams['p'], C=hparams['C'])]
hparams['test_bases_F'] = [BSplineBasis1D(h=hparams['h_F'][0], p=hparams['p'], C=hparams['C']), 
                         BSplineBasis1D(h=hparams['h_F'][1], p=hparams['p'], C=hparams['C'])]
hparams['trial_bases_F'] = [BSplineBasis1D(h=hparams['h_F'][0], p=hparams['p'], C=hparams['C']), 
                         BSplineBasis1D(h=hparams['h_F'][1], p=hparams['p'], C=hparams['C'])]

#Quadrature
hparams['quadrature'] = 'Gauss-Legendre'
hparams['n_elements'] = max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q'] = 33*hparams['n_elements']

hparams['quadrature_L'] = 'Gauss-Legendre'
hparams['n_elements_L'] = max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q_L'] = 33*hparams['n_elements_L']

#Symmetries
hparams['scaling_equivariance'] = False
hparams['permutation_equivariance'] = False

loaddir = None
logdir = '../../../nnlogs'
sublogdir = 'InvCNN_compression'
label = str(args.arg)

model = NeuralOperator
datamodule = DataModule_Darcy_MS
train(model, datamodule, hparams, loaddir, logdir, sublogdir, label)