# Copyright 2025 Joost Prins

from ngo.ml.trainer import train
from ngo.ml.customlosses import *
from ngo.ml.systemnets import *
from ngo.ml.basisfunctions import *
from ngo.testproblems.steadydiffusion.NeuralOperator import NeuralOperator
from ngo.testproblems.steadydiffusion.DataModule import DataModule

#Training data
hparams = {}
hparams['N_samples_train'] = 10 #Number of training samples
hparams['N_samples_val'] = 10 #Number of validation samples
hparams['variables'] = ['x','x'] #Variable types. 'x' for spatial variable, 't' for temporal variable
hparams['d'] = len(hparams['variables']) #Dimensionality of solution
hparams['l_min'] = [0.5,0.5] #Minimum GRF length scale per dimension
hparams['l_max'] = [1,1] #Maximum GRF length scale per dimension
hparams['gamma_stabilization'] = 0 #Stabilization constant for the system matrix

#Training settings
hparams['dtype'] = torch.float32 #Model dtype
hparams['precision'] = 32 #Keep the same as above
hparams['device'] = 'cuda:0' #Training device
hparams['devices'] = [0] #Keep number the same as the device number above
hparams['solution_loss'] = weightedrelativeL2 #For data NGO and model NGO. See src/ngo/ml/customlosses.py for the options
hparams['matrix_loss'] = None #Option: "relativematrixnorm" for data-free NGO
hparams['metric'] = weightedrelativeL2
hparams['optimizer'] = torch.optim.Adam
hparams['learning_rate'] = 1e-3
hparams['batch_size'] = 100
hparams['epochs'] = 5000

#Bases
hparams['h'] = (10,10) #Number of basis functions per dimension
hparams['p'] = (3,3) #Polynomial order per dimension (in case of B-spline basis)
hparams['C'] = (2,2) #Continuity along elements per dimension (in case of B-spline basis)
hparams['N'] = np.prod(hparams['h']) #Number of  basis degrees of freedom
hparams['trial_bases'] = [BSplineBasis1D(h=hparams['h'][0], p=hparams['p'][0], C=hparams['C'][0]), #basis functions in x
                          BSplineBasis1D(h=hparams['h'][1], p=hparams['p'][1], C=hparams['C'][1])] #basis functions in y (see src/ngo/ml/basisfunctions.py for the options)
hparams['test_bases'] = hparams['trial_bases']

#Quadrature
hparams['quadrature'] = 'Gauss-Legendre' #Quadrature rule, either "Gauss-Legendre" or "uniform" (for FNO)
hparams['n_elements'] = (3,3) #Number of elements of the quadrature grid
hparams['Q'] = (99,99) #Number of quadrature points per dimension (for all elements, not per element)
hparams['quadrature_L'] = 'Gauss-Legendre' #Loss quadrature rule, either "Gauss-Legendre" or "uniform" (for FNO)
hparams['n_elements_L'] = (3,3) #Number of elements of the loss quadrature grid
hparams['Q_L'] = (99,99) #Number of loss quadrature points per dimension (for all elements, not per element)

#System net
hparams['modeltype'] = 'model NGO' #Options: "NN" for bare NN, "DeepONet", "VarMiON", "data NGO", "model NGO"
hparams['systemnet'] = CNN #See src/ngo/ml/systemnets.py for the options
hparams['N_w'] = 30000 #Number of trainable parameters (upper bound, not exact)
hparams['skipconnections'] = True #In case of a symmetric CNN -> U-Net
hparams['kernel_sizes'] = [2,2,5,5,5,5,2,2] #In case of a CNN
hparams['bottleneck_size'] = 20 #MLP bottleneck size in case of a CNN (uses an MLP connection in the bottleneck)
hparams['outputactivation'] = nn.Tanhshrink() #For the systemnet. Hidden layer activations are ReLU by default

#Additional inductive bias
hparams['scale_equivariance'] = True #Only available for NGOs
hparams['Neumannseries'] = True #Only available for model NGOs
hparams['Neumannseries_order'] = 1 

logdir = './nnlogs' #Location for the tensorboard log files
sublogdir = 'test' #Folder name in the "nnlogs" directory
label = 'steadydiffusion_new' #Give your model a name
hparams['label'] = label

#Train the model
train(NeuralOperator, DataModule, hparams, logdir, sublogdir, label)


#Analysis of results
