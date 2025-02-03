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
hparams['project_materialparameters'] = True
hparams['project_rhs'] = False
hparams['project_u'] = False

#Training settings
hparams['dtype'] = torch.float32
hparams['precision'] = 32
hparams['discretization_device'] = 'cuda:0'
hparams['assembly_device'] = 'cpu'
hparams['train_device'] = 'cuda:0'
hparams['devices'] = [0]
hparams['solution_loss'] = weightedrelativeL2
hparams['matrix_loss'] = None #relativematrixnorm
hparams['metric'] = weightedrelativeL2
hparams['optimizer1'] = torch.optim.Adam
hparams['optimizer2'] = torch.optim.LBFGS
hparams['switch_threshold'] = None
hparams['learning_rate'] = 1e-3
hparams['batch_size'] = 100
hparams['epochs'] = 5000

#Bases
hparams['h'] = (10,10,10)
hparams['p'] = (3,3,3)
hparams['C'] = (2,2,2)
hparams['N'] = np.prod(hparams['h'])
hparams['test_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'][0], C=hparams['C'][0]),
                         BSplineBasis1D(h=hparams['h'][1], p=hparams['p'][1], C=hparams['C'][1]),
                         BSplineBasis1D(h=hparams['h'][2], p=hparams['p'][2], C=hparams['C'][2])]
# hparams['test_bases'] = [ChebyshevTBasis1D(h=hparams['h'][0]),
#                          ChebyshevTBasis1D(h=hparams['h'][1])]
# hparams['test_bases'] = [SincBasis1D(h=hparams['h'][0]),
#                          SincBasis1D(h=hparams['h'][1])]
# hparams['test_bases'] = [FourierBasis1D(h=hparams['h'][0]),
#                          FourierBasis1D(h=hparams['h'][1])]
# hparams['test_bases'] = 'POD'
hparams['trial_bases'] = hparams['test_bases']
hparams['POD'] = False

#Quadrature
hparams['quadrature'] = 'Gauss-Legendre'
hparams['n_elements'] = (3,3,3)
hparams['Q'] = (99,99,99)
hparams['quadrature_L'] = 'Gauss-Legendre'
hparams['n_elements_L'] = (3,3,3) #max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q_L'] = (99,99,99) #33*hparams['n_elements_L']

hparams['Dt'] = 1/3
hparams['n_timesteps'] = 1/hparams['Dt']

#System net'
hparams['modeltype'] = 'model NGO'
hparams['systemnet'] = CNN
# hparams['input_shape'] = (5,)+(hparams['Q'])
hparams['input_shape'] = (1,hparams['N'],hparams['N'])
# hparams['input_shape'] = (1,)+hparams['h']
# hparams['output_shape'] = hparams['h']
# hparams['output_shape'] = (hparams['Q_L'],hparams['Q_L'])
# hparams['output_shape'] = (hparams['Q_L'])
# hparams['output_shape'] = (hparams['h'][0],hparams['h'][1])
hparams['output_shape'] = (hparams['N'],hparams['N'])
hparams['N_w'] = 30000
hparams['A0net'] = None
hparams['Neumannseries'] = True
hparams['Neumannseries_order'] = 1
hparams['skipconnections'] = False
# hparams['kernel_sizes'] = [(1,1,1),(1,1,1),(2,2,2),(5,5,5),10,10,5,2]
# hparams['kernel_sizes'] = [(1,1,1),(1,1,1),(2,2,2),(5,5,5),5,2,1,1]
# hparams['kernel_sizes'] = [2,5,10,10,10,10,5,2]
# hparams['kernel_sizes'] = [10,1,2,5,5,2,1,1]
# hparams['kernel_sizes'] = [2,4,5,10,10,5,4,2]
hparams['kernel_sizes'] = [2,5,10,10,10,10,5,2]
hparams['gamma_stabilization'] = 0
hparams['bottleneck_size'] = 20
hparams['outputactivation'] = nn.Tanhshrink()

#Physics
hparams['scaling_equivariance'] = False
hparams['permutation_equivariance'] = False
hparams['massconservation'] = False

logdir = '../../../../nnlogs'
sublogdir = 'massconservation'
# sublogdir = 'test'
# hparams['N_samples_train'] = 10
# hparams['N_samples_val'] = 10
# hparams['assembly_batch_size'] = 10
label = 'modelNGO_10x10x10'
hparams['label'] = label

model = NeuralOperator
datamodule = DataModule
train(model, datamodule, hparams, loaddir, logdir, sublogdir, label)