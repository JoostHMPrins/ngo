#Standard libraries
import pytorch_lightning as pl
from tqdm.auto import tqdm, trange
from ipywidgets import IntProgress
import time
from torch import nn
from torch.nn import functional as F

#Personal ML stuff
from logger import *
from trainer import train
from customlosses import *
from systemnets import *
from basisfunctions import *

hparams = {}

#Test problem
import sys
sys.path.insert(0, '../testproblems/darcy')
from NGO import NGO
from DataModule import *

#Training data
hparams['N_samples'] = 100
hparams['d'] = 2
hparams['l_min'] = 0.5
hparams['l_max'] = 1

#Training settings
hparams['dtype'] = torch.float64
hparams['precision'] = 64
hparams['devices'] = [3]
hparams['solution_loss'] = weightedrelativeL2
hparams['matrix_loss'] = None #relativefrobeniusnorm
hparams['metric'] = weightedrelativeL2
hparams['optimizer'] = None
hparams['learning_rate'] = None
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
hparams['permutation_equivariance'] = False

loaddir = None
logdir = '../../../nnlogs'
sublogdir = 'test'
label = 'Adam+LBFGS_switchpatience=1'

model = NGO
datamodule = DataModule_Darcy_MS
train(model, datamodule, hparams, loaddir, logdir, sublogdir, label)