#Standard libraries
import pytorch_lightning as pl
from tqdm.auto import tqdm, trange
from ipywidgets import IntProgress
import time
from torch import nn
from torch.nn import functional as F

#Personal ML stuff
import sys
sys.path.insert(0, '../../ml')
from logger import *
from trainer import train
from customlosses import *
from systemnets import *
from basisfunctions import *

#Test problem
from NeuralOperator import NeuralOperator
from DataModule import DataModule

#Training data
hparams = {}
hparams['N_samples_train'] = 10000
hparams['N_samples_val'] = 1000
hparams['variables'] = ['t','x','x']
hparams['d'] = len(hparams['variables'])
hparams['l_min'] = [0.5,0.5,0.5]
hparams['l_max'] = [1,1,1]
hparams['project_materialparameters'] = False
hparams['output_coefficients'] = True

#Training settings
hparams['dtype'] = torch.float32
hparams['precision'] = 32
hparams['used_device'] = 'cuda:1'
hparams['devices'] = [1]
hparams['solution_loss'] = relativeL2_coefficients #weightedrelativeL2
hparams['matrix_loss'] = None #relativematrixnorm
hparams['metric'] = relativeL2_coefficients #weightedrelativeL2
hparams['optimizer1'] = torch.optim.Adam
hparams['optimizer2'] = torch.optim.LBFGS
hparams['switch_threshold'] = None
hparams['learning_rate'] = 1e-3
hparams['batch_size'] = 100
hparams['epochs'] = 20000

#Bases
hparams['h'] = (2,10,10)
hparams['p'] = (1,3,3)
hparams['C'] = (0,2,2)
hparams['N'] = np.prod(hparams['h'])
hparams['test_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'][0], C=hparams['C'][0]),
                         BSplineBasis1D(h=hparams['h'][1], p=hparams['p'][1], C=hparams['C'][1]),
                         BSplineBasis1D(h=hparams['h'][2], p=hparams['p'][2], C=hparams['C'][2])]
hparams['trial_bases'] = hparams['test_bases']
hparams['POD'] = False
hparams['Dt'] = 0.001

#Quadrature
hparams['quadrature'] = 'Gauss-Legendre'
hparams['n_elements'] = (1,3,3)
hparams['Q'] = (2,99,99)
hparams['quadrature_L'] = 'Gauss-Legendre'
hparams['n_elements_L'] = (1,3,3) #max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q_L'] = (2,99,99) #33*hparams['n_elements_L']

#System net'
hparams['modeltype'] = 'data NGO'
hparams['systemnet'] = CNN_3dto2d
hparams['N_w'] = 30000
hparams['bias'] = False
hparams['skipconnections'] = False
# hparams['kernel_sizes'] = [2,2,5,10,10,5,2,2]
hparams['kernel_sizes'] = [(1,1,1),(1,1,1),(1,2,2),(2,5,5),(10,10),(5,5),(2,2),(2,2)]
hparams['bottleneck_size'] = 20
hparams['outputactivation'] = None

#Additional inductive bias
hparams['gamma_stabilization'] = 0
hparams['scaling_equivariance'] = False
hparams['massconservation'] = False
hparams['Neumannseries'] = False
hparams['Neumannseries_order'] = 0
hparams['SPD'] = True
hparams['epsilon'] = 1e-4
hparams['spectral radius scaling'] = True
hparams['rho_max'] = 0.99
hparams['tol'] = 1e-4

logdir = '../../../../nnlogs'
sublogdir = 'dataNGOs_lmax'
sublogdir = 'test'
hparams['N_samples_train'] = 10
hparams['N_samples_val'] = 10
label = 'rhoscaling' 
hparams['label'] = label

model = NeuralOperator
datamodule = DataModule
train(model, datamodule, hparams, logdir, sublogdir, label)