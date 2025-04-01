# Copyright 2025 Joost Prins

from ngo.ml.trainer import train
from ngo.ml.customlosses import *
from ngo.ml.systemnets import *
from ngo.ml.basisfunctions import *
from ngo.ml.modelloader import *
from ngo.ml.quadrature import *
from ngo.testproblems.steadydiffusion.NeuralOperator import NeuralOperator
from ngo.testproblems.steadydiffusion.DataModule import DataModule
from ngo.testproblems.steadydiffusion.manufacturedsolutions import *

#Training data
hparams = {}
hparams['N_samples_train'] = 1000 #Number of training samples, default is 10000
hparams['N_samples_val'] = 100 #Number of validation samples, default is 1000
hparams['variables'] = ['x','x'] #Variable types. 'x' for spatial variable, 't' for temporal variable
hparams['d'] = len(hparams['variables']) #Dimensionality of solution
hparams['l_min'] = [0.5,0.5] #Minimum GRF length scale per dimension
hparams['l_max'] = [1,1] #Maximum GRF length scale per dimension
hparams['gamma_stabilization'] = 0 #Stabilization constant for the system matrix

#Training settings
hparams['dtype'] = torch.float32 #Model dtype
hparams['precision'] = 32 #Keep the same as above
hparams['accelerator'] = 'cpu' # 'cpu' or 'gpu'
hparams['device'] = 'cpu' #Training device
hparams['devices'] = 1 #Keep number the same as the device number above
hparams['solution_loss'] = weightedrelativeL2 #For data NGO and model NGO. See src/ngo/ml/customlosses.py for the options
hparams['matrix_loss'] = None #Option: "relativematrixnorm" for data-free NGO
hparams['metric'] = weightedrelativeL2
hparams['optimizer'] = torch.optim.Adam
hparams['learning_rate'] = 1e-3
hparams['batch_size'] = 100
hparams['epochs'] = 10 #Default for model NGO is 5000, for data NGO 20000

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
hparams['kernel_sizes'] = (2,2,5,5,5,5,2,2) #In case of a CNN
hparams['bottleneck_size'] = 20 #MLP bottleneck size in case of a CNN (uses an MLP connection in the bottleneck)
hparams['outputactivation'] = nn.Tanhshrink() #For the systemnet. Hidden layer activations are ReLU by default

#Additional inductive bias
hparams['scale_equivariance'] = True #Only available for NGOs
hparams['Neumannseries'] = True #Only available for model NGOs
hparams['Neumannseries_order'] = 1 

logdir = './nnlogs' #Location for the tensorboard log files
sublogdir = 'test' #Folder name in the "nnlogs" directory
label = 'steadydiffusion' #Give your model a name
hparams['label'] = label

#Train the model
train(NeuralOperator, DataModule, hparams, logdir, sublogdir, label)


#Plot of error versus test data length scale
print('Plotting error versus test data length scale...')
projection = loadmodelfromlabel(model=NeuralOperator, label=label, logdir=logdir, sublogdir=sublogdir, device=hparams['device'])
projection.hparams['modeltype'] = 'projection'
projection.hparams['dtype'] = torch.float64
projection.hparams['precision'] = 64
projection.__init__(projection.hparams)
modelNGO = loadmodelfromlabel(model=NeuralOperator, label=label, logdir=logdir, sublogdir=sublogdir, device=hparams['device'])
models = {'Projection': projection,
             'Model NGO': modelNGO}

N_samples = 100

alpha = 0.05
q_low = alpha/2
q_high = 1 - q_low

l = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

L2_scaled_avg = np.zeros((len(models),len(l)))
yerr = np.zeros((len(models),2,len(l)))

size_title = 20
size_ticklabels = 15
size_axeslabels = 20
linestyles=['-','--']
colors = ['black','C0']

quadrature = GaussLegendreQuadrature(n_elements=(3,3), Q=(99,99))
x = quadrature.xi
w = quadrature.w
    
for i in range(len(l)):
    print("Length scale: "+ str(l[i]))
    dataset = ManufacturedSolutionsSetDarcy(N_samples=N_samples, variables=['x','x'], l_min=[l[i],l[i]], l_max=[l[i],l[i]])
    theta = dataset.theta
    f = dataset.f
    etat = dataset.eta_yL
    etab = dataset.eta_y0
    gl = dataset.g_x0
    gr = dataset.g_xL
    u = dataset.u
    m=0
    for model in models:
        u_exact = discretize_functions(u, x)
        u_hat = models[str(model)].simforward(theta, f, etab, etat, gl, gr, x, u)
        L2_scaled_array = weightedrelativeL2_set(w, u_hat, u_exact)
        L2_scaled_avg[m,i] = np.average(L2_scaled_array)
        q_l = np.quantile(L2_scaled_array, q_low)
        q_h = np.quantile(L2_scaled_array, q_high)
        yerr[m,:,i] = np.array([L2_scaled_avg[m,i] - q_l, -L2_scaled_avg[m,i] + q_h])
        m+=1

fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

plots = []
plots.append(ax.axvspan(0.5,1, color='black', alpha=0.1, label='Training data range'))
m=0
ax.tick_params(axis='both', labelsize=size_ticklabels)
ax.set_xlabel(r'Length scale $\lambda/L$', fontsize=size_axeslabels)
ax.set_ylabel(r'Rel. L2 error $\frac{||\hat{u}-u||_2}{||u||_2}$', fontsize=size_axeslabels)
ax.set_yscale('log')
ax.set_xlim(0, 1.05)
ax.grid()
for model in models:
    plots.append(ax.errorbar(l, L2_scaled_avg[m], yerr=yerr[m], fmt=".-", capsize=6, ms=15, label=str(model), color=colors[m], linestyle=linestyles[m]))
    m+=1
plt.legend(fontsize=10, ncols=2)

plt.savefig("errorversuslengthscale.png", bbox_inches='tight')
