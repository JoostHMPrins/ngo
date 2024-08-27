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
from NeuralOperator import NeuralOperator
from DataModule import *

#Training data
hparams['N_samples'] = 100
hparams['d'] = 2
hparams['l_min'] = 0.5
hparams['l_max'] = 1

#Training settings
hparams['dtype'] = torch.float32
hparams['precision'] = 32
hparams['devices'] = [3]
hparams['solution_loss'] = weightedrelativeL2
hparams['matrix_loss'] = None #relativevectorfrobeniusnorm
hparams['metric'] = weightedrelativeL2
hparams['optimizer1'] = torch.optim.Adam
hparams['optimizer2'] = torch.optim.LBFGS
hparams['switch_threshold'] = None #5e-3
hparams['learning_rate'] = 1e-3
hparams['batch_size'] = 100
hparams['epochs'] = 5000

#Quadrature
hparams['quadrature'] = 'uniform'
hparams['n_elements'] = 1 #max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q'] = 9 #33*hparams['n_elements']
hparams['quadrature_L'] = 'uniform'
hparams['n_elements_L'] = 1 #max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q_L'] = 100 #33*hparams['n_elements_L']

#Bases
hparams['h'] = [9,9]
hparams['h_F'] = [9,9]
hparams['h_FNO'] = [5,5]
hparams['p'] = 3
hparams['C'] = 2
hparams['N'] = np.prod(hparams['h'])
hparams['N_F'] = np.prod(hparams['h_F'])
hparams['test_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'], C=hparams['C']), BSplineBasis1D(h=hparams['h'][1], p=hparams['p'], C=hparams['C'])]
hparams['trial_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'], C=hparams['C']), BSplineBasis1D(h=hparams['h'][1], p=hparams['p'], C=hparams['C'])]
hparams['test_bases_F'] = [BSplineBasis1D(h=hparams['h_F'][0], p=hparams['p'], C=hparams['C']), BSplineBasis1D(h=hparams['h_F'][1], p=hparams['p'], C=hparams['C'])]
hparams['trial_bases_F'] = [BSplineBasis1D(h=hparams['h_F'][0], p=hparams['p'], C=hparams['C']), BSplineBasis1D(h=hparams['h_F'][1], p=hparams['p'], C=hparams['C'])]

#System net'
hparams['modeltype'] = 'DeepONet'
hparams['systemnet'] = UNet
hparams['input_shape'] = (4,hparams['Q'],hparams['Q'])
hparams['gamma_stabilization'] = 0
hparams['lastskip'] = False
hparams['kernel_sizes'] = [1,1,3,3,3,3,1,9]
hparams['N_w'] = 30000
hparams['bottleneck_size'] = 20
hparams['NLB_outputactivation'] = None

#Symmetries
hparams['scaling_equivariance'] = False
hparams['permutation_equivariance'] = False

loaddir = None
logdir = '../../../nnlogs'
sublogdir = 'test'
label = 'DeepONet'

model = NeuralOperator
datamodule = DataModule_Darcy_MS
train(model, datamodule, hparams, loaddir, logdir, sublogdir, label)