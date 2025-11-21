# Copyright 2025 Joost Prins

# 3rd Party
import torch

# Local
from ngo.ml.trainer import train
from ngo.ml.customlosses import *
from ngo.ml.systemnets import *
from ngo.numericalanalysis.basisfunctions import *
from ngo.testproblems.steadydiffusion.NeuralOperator import NeuralOperator
from ngo.testproblems.steadydiffusion.DataModule import DataModule

#Data type and device
hparams = {}
hparams['dtype'] = torch.float32
hparams['device'] = 'cuda'

#Training data
hparams['N_samples_train'] = 100 #Number of training samples
hparams['N_samples_val'] = 10 #Number of validation samples
hparams['l_min'] = torch.tensor([0.5,0.5]) #Minimum GRF length scale per dimension
hparams['l_max'] = torch.tensor([1,1]) #Maximum GRF length scale per dimension
hparams['d'] = len(hparams['l_min']) #Number of spatial dimensions
hparams['b_theta_min'] = 1 #minimum coefficient field mean
hparams['b_theta_max'] = 1 #maximum coefficient field mean
hparams['c_theta_min'] = 0 #minimum coefficient field standard deviation
hparams['c_theta_max'] = 0.2 #maximum coefficient field standard deviation
hparams['b_u_min']  = -1 #minimum solution field mean   
hparams['b_u_max']  = 1 #maximum solution field mean
hparams['c_u_min']  = 0 #minimum solution field standard deviation
hparams['c_u_max']  = 1 #maximum solution field standard deviation
hparams['n_RBF'] = (3,3) #Number of basis functions per dimension for data generation

#Training settings
hparams['precision'] = 32 #Keep the same as above
hparams['device'] = 'cuda' #Training device
hparams['solution_loss'] = weightedrelativeL2 #For data NGO and model NGO
hparams['matrix_loss'] = None #Option: "relativematrixnorm" for data-free NGO
hparams['metric'] = weightedrelativeL2
hparams['optimizer'] = torch.optim.Adam
hparams['learning_rate'] = 1e-3
hparams['batch_size'] = 100
hparams['epochs'] = 10

#Bases
hparams['N_i'] = (10,10) #Number of basis functions per dimension
hparams['p'] = (3,3) #Polynomial order per dimension (in case of B-spline basis)
hparams['C'] = (2,2) #Continuity along elements per dimension (in case of B-spline basis)
hparams['N'] = np.prod(hparams['n']) #Number of  basis degrees of freedom
hparams['test_bases'] = [BSplineBasis1D(n=hparams['N_i'][0], p=hparams['p'][0], C=hparams['C'][0]),
                         BSplineBasis1D(n=hparams['N_i'][1], p=hparams['p'][1], C=hparams['C'][1])] #See basisfunctions.py for the options
hparams['trial_bases'] = hparams['test_bases']

#Quadrature
hparams['quadrature'] = 'Gauss-Legendre' #Quadrature rule, either "Gauss-Legendre" or "uniform" (for FNO)
hparams['n_elements'] = (3,3) #Number of elements of the quadrature grid
hparams['Q'] = (99,99) #Number of quadrature points per dimension (for all elements, not per element)
hparams['quadrature_L'] = 'Gauss-Legendre' #Loss quadrature rule, either "Gauss-Legendre" or "uniform" (for FNO)
hparams['n_elements_L'] = (3,3) #Number of elements of the loss quadrature grid
hparams['Q_L'] = (99,99) #Number of loss quadrature points per dimension (for all elements, not per element)

#System net
hparams['modeltype'] = 'model NGO' #Options: "NN" for bare NN, "DeepONet", "VarMiON", "data NGO", "model NGO"
hparams['systemnet'] = CNN #See systemnets.py for the options
hparams['N_w'] = 30000 #Number of trainable parameters (upper bound, not exact)
hparams['skipconnections'] = True #In case of a symmetric CNN -> U-Net
hparams['kernel_sizes'] = [2,2,5,5,5,5,2,2] #In case of a CNN
# hparams['kernel_sizes'] = [10,1,2,5,5,2,1,1] #In case of a CNN
# hparams['kernel_sizes'] = [2,2,5,5,5,5,2,2] #In case of a CNN
hparams['bottleneck_size'] = 20 #MLP bottleneck size in case of a CNN (uses an MLP connection in the bottleneck)
hparams['outputactivation'] = nn.Tanhshrink() #For the systemnet

#Inductive bias
hparams['gamma_stabilization'] = 0 #Stabilization constant for the system matrix
hparams['Neumannseries'] = True
hparams['Neumannseries_order'] = 1
hparams['scale_equivariance'] = False

logdir = '../../../../../../nnlogs' #Location for the tensorboard log files
sublogdir = 'testsnellius' #Folder name in the "logs" directory
# sublogdir = 'test'
# hparams['N_samples_train'] = 10
# hparams['N_samples_val'] = 10
label = 'modelNGO_Neumann2' #Give your model a name
hparams['label'] = label

model = NeuralOperator
datamodule = DataModule
train(model, datamodule, hparams, logdir, sublogdir, label)