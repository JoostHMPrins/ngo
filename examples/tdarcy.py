# Copyright 2025 Joost Prins

# Local Utilities
from ngo.ml.logger import *
from ngo.ml.trainer import train
from ngo.ml.customlosses import *
from ngo.ml.systemnets import *
from ngo.ml.basisfunctions import *

#Test problem
from ngo.testproblems.tdarcy.NeuralOperator import NeuralOperator
from ngo.testproblems.tdarcy.DataModule import DataModule

#Training data
hparams = {}
hparams['N_samples_train'] = 100#10000
hparams['N_samples_val'] = 10#00
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
hparams['epochs'] = 5000

#Bases
hparams['h'] = (1,10,10)
hparams['p'] = (0,3,3)
hparams['C'] = (-1,2,2)
hparams['N'] = np.prod(hparams['h'])
hparams['test_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'][0], C=hparams['C'][0]),
                         BSplineBasis1D(h=hparams['h'][1], p=hparams['p'][1], C=hparams['C'][1]),
                         BSplineBasis1D(h=hparams['h'][2], p=hparams['p'][2], C=hparams['C'][2])]
hparams['trial_bases'] = hparams['test_bases']
hparams['POD'] = False
hparams['Dt'] = 0.02

#Quadrature
hparams['quadrature'] = 'Gauss-Legendre'
hparams['n_elements'] = (1,3,3)
hparams['Q'] = (2,99,99)
hparams['quadrature_L'] = 'Gauss-Legendre'
hparams['n_elements_L'] = (1,3,3) #max(int((hparams['h'][0] - 1)/hparams['p']), 1)
hparams['Q_L'] = (2,99,99) #33*hparams['n_elements_L']

#System net'
hparams['modeltype'] = 'model NGO'
hparams['systemnet'] = CNN
hparams['N_w'] = 30000
hparams['skipconnections'] = True
# hparams['kernel_sizes'] = [(1,1,1),(1,1,1),(2,2,2),(5,5,5),10,10,5,2]
# hparams['kernel_sizes'] = [(1,1,1),(1,1,1),(2,2,2),(5,5,5),5,2,1,1]
hparams['kernel_sizes'] = [2,2,5,5,5,5,2,2]
# hparams['kernel_sizes'] = [10,1,2,5,5,2,1,1]
# hparams['kernel_sizes'] = [2,4,5,5,5,5,4,2]
# hparams['kernel_sizes'] = [(1,1,1),(1,1,1),(1,1,1),(2,5,5),5,5,4,2]
# hparams['kernel_sizes'] = [(1,1,1),(1,2,2),(1,3,3),(3,5,5),5,5,4,2]

hparams['gamma_stabilization'] = 0
hparams['bottleneck_size'] = 20
hparams['outputactivation'] = None

#Addiional inductive bias
hparams['scaling_equivariance'] = False
hparams['massconservation'] = False
hparams['Neumannseries'] = True
hparams['Neumannseries_order'] = 1

logdir = '../../../../nnlogs'
sublogdir = 'tdd2'
# sublogdir = 'test'
hparams['N_samples_train'] = 10
hparams['N_samples_val'] = 10
hparams['assembly_batch_size'] = 10
label = 'modelNGO'
hparams['label'] = label

model = NeuralOperator
datamodule = DataModule
train(model, datamodule, hparams, logdir, sublogdir, label)