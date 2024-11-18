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
hparams['N_samples_train'] = 100
hparams['N_samples_val'] = 100
hparams['d'] = 2
hparams['l_min'] = 0.5
hparams['l_max'] = 1
loaddir = None #'../../../trainingdata/darcy_mfs_l1e-2to1e0'

#Training settings
hparams['dtype'] = torch.float32
hparams['precision'] = 32
hparams['devices'] = [0]
hparams['used_device'] = 'cuda:0'
hparams['solution_loss'] = weightedrelativeL2
hparams['matrix_loss'] = None #relativematrixnorm
hparams['metric'] = weightedrelativeL2
hparams['optimizer1'] = torch.optim.Adam
hparams['optimizer2'] = torch.optim.LBFGS
hparams['switch_threshold'] = None
hparams['learning_rate'] = 1e-3
hparams['batch_size'] = 100
hparams['epochs'] = 100

#Bases
hparams['h'] = [10,10]
hparams['p'] = 3
hparams['C'] = 2
hparams['N'] = np.prod(hparams['h'])
# hparams['test_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'], C=hparams['C']), BSplineBasis1D(h=hparams['h'][1], p=hparams['p'], C=hparams['C'])]
# hparams['trial_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'], C=hparams['C']), BSplineBasis1D(h=hparams['h'][1], p=hparams['p'], C=hparams['C'])]
# hparams['test_bases'] = [ChebyshevTBasis1D(h=hparams['h'][0]), ChebyshevTBasis1D(h=hparams['h'][1])]
# hparams['trial_bases'] = [ChebyshevTBasis1D(h=hparams['h'][0]), ChebyshevTBasis1D(h=hparams['h'][1])]
hparams['test_bases'] = [SincBasis1D(h=hparams['h'][0]), ChebyshevTBasis1D(h=hparams['h'][1])]
hparams['trial_bases'] = [SincBasis1D(h=hparams['h'][0]), ChebyshevTBasis1D(h=hparams['h'][1])]

#Quadrature
hparams['quadrature'] = 'Gauss-Legendre'
hparams['n_elements'] = 3 #max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q'] = 99 #33*hparams['n_elements']
hparams['quadrature_L'] = 'uniform'
hparams['n_elements_L'] = 1 #max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q_L'] = 100 #33*hparams['n_elements_L']

#System net'
hparams['modeltype'] = 'model NGO'
hparams['systemnet'] = CNN
hparams['input_shape'] = (1,hparams['N'],hparams['N'])
hparams['output_shape'] = (hparams['N'],hparams['N'])
hparams['N_w'] = 30000
hparams['A0net'] = None
hparams['Neumannseries'] = False #True
hparams['Neumannseries_order'] = None #1
hparams['skipconnections'] = True
hparams['kernel_sizes'] = [2,2,5,5,5,5,2,2]
hparams['gamma_stabilization'] = 0
hparams['bottleneck_size'] = 20
hparams['outputactivation'] = None #nn.Tanhshrink()

#Symmetries
hparams['scaling_equivariance'] = True
hparams['permutation_equivariance'] = False

logdir = '../../../nnlogs'
# sublogdir = 'test'
# hparams['N_samples_train'] = 1
# hparams['N_samples_val'] = 1
label = 'sinc'
hparams['label'] = label

model = NeuralOperator
datamodule = DataModule_Darcy_MS
train(model, datamodule, hparams, loaddir, logdir, sublogdir, label)