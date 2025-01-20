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
sys.path.insert(0, '../testproblems/tdarcy')
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
loaddir = None #'../../../trainingdata/darcy_mfs_l1e-2to1e0'
hparams['project_materialparameters'] = False
hparams['project_rhs'] = False
hparams['assembly_batch_size'] = 1000

#Training settings
hparams['dtype'] = torch.float
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
hparams['epochs'] = 20000

#Bases
hparams['h'] = [10,10,10]
hparams['p'] = [3,3,3]
hparams['C'] = [2,2,2]
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
hparams['n_elements'] = [3,3,3]
hparams['Q'] = [99,99,99]
hparams['quadrature_L'] = 'Gauss-Legendre'
hparams['n_elements_L'] = [3,3,3] #max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q_L'] = [12,12,12] #33*hparams['n_elements_L']

#System net'
hparams['modeltype'] = 'data NGO'
hparams['systemnet'] = CNNNd
# hparams['input_shape'] = (1,hparams['Q'],hparams['Q'])
hparams['input_shape'] = (1,hparams['h'][0],hparams['h'][1],hparams['h'][2])
# hparams['input_shape'] = (1,hparams['N'],hparams['N'])
# hparams['output_shape'] = (hparams['Q_L'],hparams['Q_L'])
hparams['output_shape'] = (hparams['N'],hparams['N'])
# hparams['output_shape'] = (hparams['h'][0],hparams['h'][1])
hparams['N_w'] = 30000
hparams['A0net'] = None
hparams['Neumannseries'] = False
hparams['Neumannseries_order'] = None
hparams['skipconnections'] = False
hparams['kernel_sizes'] = [(1,1,1),(1,1,1),(2,2,2),(5,5,5),10,10,5,2]
# hparams['kernel_sizes'] = [2,5,10,10,10,10,5,2]
# hparams['kernel_sizes'] = [10,1,2,5,5,2,1,1]
# hparams['kernel_sizes'] = [2,5,10,10,10,10,5,2]
hparams['gamma_stabilization'] = 0
hparams['bottleneck_size'] = 20
hparams['outputactivation'] = None #nn.Tanhshrink()

#Symmetries
hparams['scaling_equivariance'] = False
hparams['permutation_equivariance'] = False

logdir = '../../../nnlogs'
sublogdir = 'tdarcy'
sublogdir = 'test'
hparams['assembly_batch_size'] = 1
hparams['N_samples_train'] = 1
hparams['N_samples_val'] = 1
label = 'dataNGO'
hparams['label'] = label

model = NeuralOperator
datamodule = DataModule
train(model, datamodule, hparams, loaddir, logdir, sublogdir, label)