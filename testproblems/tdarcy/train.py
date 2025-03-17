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
hparams['assembly_batch_size'] = 1100
hparams['variables'] = ['t','x','x']
hparams['d'] = len(hparams['variables'])
hparams['l_min'] = [0.5,0.5,0.5]
hparams['l_max'] = [1,1,1]
loaddir = None #'../../../trainingdata/darcy_mfs_l1e-2to1e0'
hparams['project_materialparameters'] = False
hparams['output_coefficients'] = True

#Training settings
hparams['dtype'] = torch.float32
hparams['precision'] = 32
hparams['used_device'] = 'cuda:0'
hparams['devices'] = [0]
hparams['solution_loss'] = relativeL2_coefficients
hparams['matrix_loss'] = None #relativematrixnorm
hparams['metric'] = relativeL2_coefficients
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
hparams['Dt'] = 0.01

#Quadrature
hparams['quadrature'] = 'Gauss-Legendre'
hparams['n_elements'] = (1,3,3)
hparams['Q'] = (3,30,30)
hparams['quadrature_L'] = 'Gauss-Legendre'
hparams['n_elements_L'] = (1,3,3) #max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q_L'] = (3,30,30) #33*hparams['n_elements_L']

#System net'
hparams['modeltype'] = 'VarMiON'
hparams['systemnet'] = CNN_3dto2d
hparams['N_w'] = 1000000
hparams['skipconnections'] = False
# hparams['kernel_sizes'] = [(1,1,1),(1,1,1),(2,2,2),(5,5,5),10,10,5,2]
# hparams['kernel_sizes'] = [(1,1,1),(1,1,1),(2,2,2),(5,5,5),5,2,1,1]
# hparams['kernel_sizes'] = [2,5,10,10,10,10,5,2]
# hparams['kernel_sizes'] = [10,1,2,5,5,2,1,1]
# hparams['kernel_sizes'] = [2,4,5,5,5,5,4,2]
# hparams['kernel_sizes'] = [(1,1,1),(1,1,1),(1,1,1),(2,5,5),5,5,4,2]
hparams['kernel_sizes'] = [(1,1,1),(1,2,2),(1,3,3),(3,5,5),5,5,4,2]

hparams['gamma_stabilization'] = 0
hparams['bottleneck_size'] = 20
hparams['outputactivation'] = None

#Addiional inductive bias
hparams['scaling_equivariance'] = False
hparams['massconservation'] = False
hparams['Neumannseries'] = False #True
hparams['Neumannseries_order'] = None

logdir = '../../../../nnlogs'
sublogdir = 'tdd'
# sublogdir = 'test'
# hparams['N_samples_train'] = 10
# hparams['N_samples_val'] = 10
# hparams['assembly_batch_size'] = 10
label = 'VarMiON_GL_Nw1e6'
hparams['label'] = label

model = NeuralOperator
datamodule = DataModule
train(model, datamodule, hparams, loaddir, logdir, sublogdir, label)