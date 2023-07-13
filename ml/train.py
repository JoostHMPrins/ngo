import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm.auto import tqdm, trange
from ipywidgets import IntProgress
import time
from torch.nn import functional as F
from torchmetrics.functional.regression import mean_absolute_percentage_error

import sys
sys.path.insert(0, '../ml')
from VarMiON import *
from DataModule import *
from logger import *
from trainer import train

hparams = {'accelerator': 'gpu',
           'devices': [0,1],
           'loss_terms': [F.mse_loss],
           'loss_coeffs': [1],
           'metric': mean_absolute_percentage_error,
           'optimizer': torch.optim.Adam, 
           'learning_rate': 1e-1,
           'batch_size': 100, 
           'max_epochs': 1000,
           'early_stopping_patience': 1000000,
           'dtype': torch.float,
           'precision': 32}

loaddir = '../../../trainingdata/polynomial_large'
logdir = '../../../nnlogs'
sublogdir = 'polynomial'
label = 'lr1e-1'

model = VarMiON

train(model, hparams, loaddir, logdir, sublogdir, label)