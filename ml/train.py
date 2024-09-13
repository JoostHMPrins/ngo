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

#Test problem
import sys
sys.path.insert(0, '../testproblems/darcy')
from NeuralOperator import NeuralOperator
from DataModule import *

#Training data
hparams = {}
hparams['N_samples'] = 1000
hparams['d'] = 2
hparams['l_min'] = 0.5
hparams['l_max'] = 1

#Training settings
hparams['dtype'] = torch.float32
hparams['precision'] = 32
hparams['devices'] = [1]
hparams['used_device'] = 'cuda:1'
hparams['solution_loss'] = weightedrelativeL2
hparams['matrix_loss'] = None #relativematrixnorm
hparams['metric'] = weightedrelativeL2
hparams['optimizer1'] = torch.optim.Adam
hparams['optimizer2'] = torch.optim.LBFGS
hparams['switch_threshold'] = 1e-2
hparams['learning_rate'] = 1e-3
hparams['batch_size'] = 100
hparams['epochs'] = 50000

#Bases
hparams['h'] = [10,10]
hparams['h_F'] = [10,10]
hparams['h_FNO'] = [5,5]
hparams['p'] = 3
hparams['C'] = 2
hparams['N'] = np.prod(hparams['h'])
hparams['N_F'] = np.prod(hparams['h_F'])
hparams['test_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'], C=hparams['C']), BSplineBasis1D(h=hparams['h'][1], p=hparams['p'], C=hparams['C'])]
hparams['trial_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'], C=hparams['C']), BSplineBasis1D(h=hparams['h'][1], p=hparams['p'], C=hparams['C'])]
hparams['test_bases_F'] = [BSplineBasis1D(h=hparams['h_F'][0], p=hparams['p'], C=hparams['C']), BSplineBasis1D(h=hparams['h_F'][1], p=hparams['p'], C=hparams['C'])]
hparams['trial_bases_F'] = [BSplineBasis1D(h=hparams['h_F'][0], p=hparams['p'], C=hparams['C']), BSplineBasis1D(h=hparams['h_F'][1], p=hparams['p'], C=hparams['C'])]

#Quadrature
hparams['quadrature'] = 'Gauss-Legendre'
hparams['n_elements'] = max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q'] = 33*hparams['n_elements']
hparams['quadrature_L'] = 'Gauss-Legendre'
hparams['n_elements_L'] = max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q_L'] = 33*hparams['n_elements_L']

#System net'
hparams['modeltype'] = 'model NGO'
hparams['systemnet'] = InvNet
hparams['input_shape'] = (1,hparams['N'],hparams['N'])
hparams['output_shape'] = (hparams['N'],hparams['N'])
hparams['kernel_sizes'] = [1,1,1,5,5,1,1,1]
hparams['gamma_stabilization'] = 0
hparams['lastskip'] = None
hparams['N_w'] = 30000
hparams['bottleneck_size'] = 20
hparams['NLB_outputactivation'] = None

#Symmetries
hparams['scaling_equivariance'] = False
hparams['permutation_equivariance'] = False

loaddir = None
logdir = '../../../nnlogs'
sublogdir = 'InvNet_c5b20'
label = 'Adam+LBFGS'
hparams['label'] = label


model = NeuralOperator
datamodule = DataModule_Darcy_MS
train(model, datamodule, hparams, loaddir, logdir, sublogdir, label)