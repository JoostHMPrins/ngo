import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import torch
import os

import sys
from manufacturedsolutions import *
sys.path.insert(0, '../../ml')
from modelloader import loadmodelfromlabel
from NeuralOperator import NeuralOperator
from customlosses import *
from quadrature import *
from customlayers import discretize_functions
from basisfunctions import *



from NeuralOperator import NeuralOperator

logdir = '../../../../nnlogs'
device = 'cuda:2'

sublogdir = 'tdd'

projection = loadmodelfromlabel(model=NeuralOperator, label='modelNGO', logdir=logdir, sublogdir=sublogdir, device=device)
projection.hparams['modeltype'] = 'projection'
projection.hparams['dtype'] = torch.float64
projection.__init__(projection.hparams)

FEM = loadmodelfromlabel(model=NeuralOperator, label='modelNGO', logdir=logdir, sublogdir=sublogdir, device=device)
FEM.hparams['modeltype'] = 'FEM'
FEM.hparams['dtype'] = torch.float64
FEM.hparams['Neumannseries'] = False
FEM.hparams['Neumannseries_order'] = None
FEM.hparams['project_materialparameters'] = True
FEM.hparams['gamma_stabilization'] = 100
FEM.__init__(FEM.hparams)

NN = loadmodelfromlabel(model=NeuralOperator, label='NN', logdir=logdir, sublogdir=sublogdir, device=device)

FNO = loadmodelfromlabel(model=NeuralOperator, label='FNO', logdir=logdir, sublogdir=sublogdir, device=device)

DeepONet = loadmodelfromlabel(model=NeuralOperator, label='DeepONet', logdir=logdir, sublogdir=sublogdir, device=device)

VarMiON = loadmodelfromlabel(model=NeuralOperator, label='VarMiON2', logdir=logdir, sublogdir=sublogdir, device=device)

dataNGO = loadmodelfromlabel(model=NeuralOperator, label='dataNGO', logdir=logdir, sublogdir=sublogdir, device=device)

datafreeNGO = loadmodelfromlabel(model=NeuralOperator, label='datafreeNGO', logdir=logdir, sublogdir=sublogdir, device=device)

modelNGO = loadmodelfromlabel(model=NeuralOperator, label='modelNGO', logdir=logdir, sublogdir=sublogdir, device=device)

modeldict = {
            'Projection': projection,
            # 'Stabilized FEM': FEM,
             'NN': NN,
             'FNO': FNO,
             'DeepONet': DeepONet,
             'VarMiON': VarMiON,
            'Data NGO': dataNGO,
            'Data-free NGO': datafreeNGO,
            'Model NGO': modelNGO,
}


N_samples = 100
n_timesteps = 1000
t_values = np.linspace(0,n_timesteps*projection.hparams['Dt'],n_timesteps)

alpha = 0.05
q_low = alpha/2
q_high = 1 - q_low

L2_scaled_avg = np.zeros((len(modeldict),n_timesteps))
yerr = np.zeros((len(modeldict),2,n_timesteps))

size_title = 20
size_ticklabels = 15
size_axeslabels = 20

quadrature = UniformQuadrature(Q=[2,100,100])
# quadrature = GaussLegendreQuadrature(n_elements=(1,3,3),Q=[2,99,99])
x = quadrature.xi
w = quadrature.w
w = w*projection.hparams['Dt']
# x[:,0] = 0
# linestyles=['-','-','--','--','--','--','--','--','--','--','--','--','--','--','--']
# colors = ['black','grey','C0','C1','C2','C3','C4','C5','C6','C7','C8']

# linestyles=['-','-','--','--','--','--','--']
# colors = ['black','grey','red','orange','green']

# linestyles=['--','--','--','--','--','--']
# colors = ['red','green']

linestyles=['-','dotted','dotted','--','--','--','--','--','--','--']
colors = ['black','C3','C6','C4','C5','C0','C1','C2','C7']

dataset = ManufacturedSolutionsSet(N_samples=N_samples, variables=['t','x','x'], l_min=[0.1,0.1,0.1], l_max=[0.1,0.1,0.1])
theta = dataset.theta
f = dataset.f
etat = dataset.etat
etab = dataset.etab
gl = dataset.gl
gr = dataset.gr
u = dataset.u
u0 = dataset.u0

m=0
for model in modeldict:
    print(str(model))
    modeldict[str(model)].timestep = 0
    x_scaled = np.copy(x)
    x_scaled[:,0] = x_scaled[:,0]*modeldict[str(model)].hparams['Dt']
    for i in range(n_timesteps):
        u_exact = discretize_functions(u, x_scaled)
        # u_exact = projection.simforward(theta, f, etab, etat, gl, gr, u0, x, u)
        u_hat = modeldict[str(model)].simforward(theta, f, etab, etat, gl, gr, u0, x, u)
        # c = modeldict[str(model)].compute_stability_criterion(theta)
        # print(np.amax(c))
        # print(np.amin(c))
        L2_scaled_array = weightedrelativeL2_set(w, u_hat, u_exact)
        L2_scaled_avg[m,i] = np.average(L2_scaled_array)
        q_l = np.quantile(L2_scaled_array, q_low)
        q_h = np.quantile(L2_scaled_array, q_high)
        yerr[m,:,i] = np.array([q_l, q_h])
        modeldict[str(model)].translate_quadrature_t()
        x_scaled[:,0] = x_scaled[:,0] + modeldict[str(model)].hparams['Dt']
        modeldict[str(model)].timestep += 1
        print(i)
    m+=1

fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

plots = []
m=0
ax.tick_params(axis='both', labelsize=size_ticklabels)
ax.set_xlabel(r'Time $t/T$', fontsize=size_axeslabels)
# ax.set_ylabel(r'Rel. L2 error $\frac{||\hat{u}-u^{\mathrm{p}}||_2}{||u^{\mathrm{p}}||_2}$', fontsize=size_axeslabels)
ax.set_ylabel(r'Rel. L2 error $\frac{||\hat{u}-u||_2}{||u||_2}$', fontsize=size_axeslabels)

ax.set_yscale('log')
# ax.set_xlim(0, 1)
ax.set_ylim(1e-2,1e2)

ax.set_yticks([1e-2,1e-1,1e0,1e1,1e2])
ax.grid()
for model in modeldict:
    # plots.append(ax.errorbar(t_values, L2_scaled_avg[m], yerr=yerr[m], fmt=".-", capsize=6, ms=15, label=str(model), color=colors[m], linestyle=linestyles[m]))
    plots.append(ax.plot(t_values, L2_scaled_avg[m], label=str(model), color=colors[m], linestyle=linestyles[m], linewidth=3)[0])
    ax.fill_between(t_values, yerr[m,0,:], yerr[m,1,:], alpha=0.1, color=colors[m])
    m+=1
plt.legend(fontsize=10, loc='upper right', ncols=2)
# legend1 = ax.legend(handles=plots[0:1],fontsize=10, ncols=1, loc='lower left')
# plt.gca().add_artist(legend1)
# legend2 = ax.legend(handles=plots[1:],fontsize=10, ncols=2, loc='lower right')
# plt.gca().add_artist(legend2)

plt.savefig("errorsovertime.svg", bbox_inches='tight')#, transparent=True)

plt.show()