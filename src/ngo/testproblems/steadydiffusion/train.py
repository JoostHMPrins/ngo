# Copyright (c) 2023-2025 Eindhoven University of Technology

# 3rd Party
import torch

# Local
from ngo.ml.trainer import train
from ngo.ml.customlosses import *
from ngo.ml.systemnets import *
from ngo.numericalanalysis.basisfunctions import *
from ngo.testproblems.steadydiffusion.NeuralOperator import NeuralOperator
from ngo.testproblems.steadydiffusion.DataModule import DataModule

hparams = {}

#Data type and device
hparams['dtype'] = torch.float64 #Model dtype
hparams['accelerator'] = 'gpu' # Can be set to 'cpu' if desired
hparams['device'] = 'cuda:0' # Can be set to 'cpu' if desired

#Training data
hparams['N_samples_train'] = 100 #Number of training samples
hparams['N_samples_val'] = 10 #Number of validation samples
hparams['l_min'] = torch.tensor([0.5,0.5], dtype=hparams['dtype'], device=hparams['device']) #Minimum GRF length scale per dimension
hparams['l_max'] = torch.tensor([1,1], dtype=hparams['dtype'], device=hparams['device']) #Maximum GRF length scale per dimension
hparams['d'] = len(hparams['l_min']) #Number of spatial dimensions
hparams['b_theta_min'] = 1 #minimum coefficient field mean
hparams['b_theta_max'] = 1 #maximum coefficient field mean
hparams['c_theta_min'] = 0 #minimum coefficient field standard deviation
hparams['c_theta_max'] = 0.2 #maximum coefficient field standard deviation
hparams['b_u_min']  = -1 #minimum solution field mean   
hparams['b_u_max']  = 1 #maximum solution field mean
hparams['c_u_min']  = 0 #minimum solution field standard deviation
hparams['c_u_max']  = 1 #maximum solution field standard deviation
hparams['n_RBF'] = (5,5) #Number of basis functions per dimension for data generation

#Training settings
hparams['solution_loss'] = relativeL2norm #relativeL2norm #For data NGO and model NGO. See src/ngo/ml/customlosses.py for the options
hparams['matrix_loss'] = None #relativematrixnorm #Option: "relativematrixnorm" for data-free NGO
hparams['metric'] = relativeL2norm
hparams['optimizer'] = torch.optim.Adam
hparams['learning_rate'] = 1e-3
hparams['batch_size'] = 100
hparams['epochs'] = 20000 #Default for model NGO is 5000, for data NGO 20000

#Bases
hparams['N_i'] = (10,10) #Number of basis functions per dimension
hparams['p'] = (3,3) #Polynomial order per dimension (in case of B-spline basis)
hparams['C'] = (2,2) #Continuity along elements per dimension (in case of B-spline basis)
hparams['N'] = np.prod(hparams['N_i']) #Number of  basis degrees of freedom
hparams['trial_bases'] = [BSplineBasis(N=hparams['N_i'][0], p=hparams['p'][0], C=hparams['C'][0], dtype=hparams['dtype'], device=hparams['device']), #basis functions in x
                          BSplineBasis(N=hparams['N_i'][1], p=hparams['p'][1], C=hparams['C'][1], dtype=hparams['dtype'], device=hparams['device'])] #basis functions in y (see src/ngo/ml/basisfunctions.py for the options)
hparams['test_bases'] =  [BSplineBasis(N=hparams['N_i'][0], p=hparams['p'][0], C=hparams['C'][0], dtype=hparams['dtype'], device=hparams['device']), #basis functions in x
                          BSplineBasis(N=hparams['N_i'][1], p=hparams['p'][1], C=hparams['C'][1], dtype=hparams['dtype'], device=hparams['device'])] #basis functions in y (see src/ngo/ml/basisfunctions.py for the options)

#Quadrature
hparams['quadrature'] = 'uniform' #Quadrature rule, either "Gauss-Legendre" or "uniform" (for FNO)
hparams['n_elements'] = (3,3) #Number of elements of the quadrature grid
hparams['Q'] = (100,100) #Number of quadrature points per dimension (for all elements, not per element)
hparams['quadrature_L'] = 'uniform' #Loss quadrature rule, either "Gauss-Legendre" or "uniform" (for FNO)
hparams['n_elements_L'] = (3,3) #Number of elements of the loss quadrature grid
hparams['Q_L'] = (100,100) #Number of loss quadrature points per dimension (for all elements, not per element)

#System net
hparams['modeltype'] = 'NN' #Options: "NN" for bare NN, "DeepONet", "VarMiON", "data NGO", "model NGO"
hparams['systemnet'] = CNN #See src/ngo/ml/systemnets.py for the options
hparams['N_w'] = 30000 #Number of trainable parameters (upper bound, not exact)
hparams['skipconnections'] = True #In case of a symmetric CNN -> U-Net
# hparams['kernel_sizes'] = (2,2,5,5,5,5,2,2) #In case of a CNN
hparams['kernel_sizes'] = (2,2,5,5,5,5,2,2) #In case of a CNN
hparams['bottleneck_size'] = 20 #MLP bottleneck size in case of a CNN (uses an MLP connection in the bottleneck)
hparams['outputactivation'] = None #nn.Tanhshrink() #For the systemnet. Hidden layer activations are ReLU by default

#Inductive bias
hparams['output_coefficients'] = False #If True, the model outputs basis coefficients. If False, the model outputs solution values at quadrature points.
hparams['gamma_stabilization'] = 0 #Stabilization constant for the system matrix
hparams['scale_equivariance'] = True #Only available for NGOs
hparams['Neumannseries'] = False #Only available for model NGOs
hparams['Neumannseries_order'] = 1 

logdir = '../../../../../../nnlogs' #Location for the tensorboard log files
sublogdir = 'test' #Folder name in the "nnlogs" directory
label = 'modelNGO' #Give your model a name
hparams['label'] = label

model = NeuralOperator
datamodule = DataModule
train(model, datamodule, hparams, logdir, sublogdir, label)