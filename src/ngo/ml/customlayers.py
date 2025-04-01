# Copyricht 2025 Joost Prins

# 3rd Party
import torch
import numpy as np
from torch import nn
from joblib import Parallel, delayed
    

class ReshapeLayer(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape
    
    def forward(self, x):
        new_shape = (x.shape[0],) + self.output_shape
        y = x.reshape(new_shape)
        return y
    

def discretize_functions(f_list, x):
    f_discretized = Parallel(n_jobs=-1)(delayed(f)(x) for f in f_list)
    f_discretized = np.array(f_discretized)
    return f_discretized


def balance_num_trainable_params(model, N_w):
    model.free_parameter = 1
    N_w_real_0 = model.hparams['N_w_real']
    free_parameter_values_list = []
    N_w_real = N_w_real_0
    while N_w_real < N_w:
        model.init_layers()
        N_w_real = N_w_real_0 + sum(p.numel() for p in model.parameters())
        free_parameter_values_list.append(model.free_parameter)
        model.free_parameter +=1
    if len(free_parameter_values_list)>1:
        model.free_parameter = free_parameter_values_list[-2]
    else:
        model.free_parameter = free_parameter_values_list[0]
    model.init_layers()
    count = sum(p.numel() for p in model.parameters())
    model.hparams['N_w_real'] += count
    if model.hparams['N_w_real']>N_w:
        raise ValueError('Not enough trainable parameter budget.')
    return model  