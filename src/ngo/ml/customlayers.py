# Copyricht 2025 Joost Prins

# 3rd Party
import torch
import numpy as np
from torch import nn
from joblib import Parallel, delayed
    

class ReshapeLayer(nn.Module):
    """
    A custom PyTorch layer to reshape the input tensor to a specified output shape.

    Attributes:
        output_shape (tuple): The desired shape of the output tensor (excluding the batch dimension).
    """
    def __init__(self, output_shape):
        """
        Initialize the ReshapeLayer.

        Args:
            output_shape (tuple): The desired shape of the output tensor (excluding the batch dimension).
        """
        super().__init__()
        self.output_shape = output_shape
    
    def forward(self, x):
        """
        Reshape the input tensor to the specified output shape.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, ...).

        Returns:
            torch.Tensor: Reshaped tensor. Shape: (batch_size, *output_shape).
        """
        new_shape = (x.shape[0],) + self.output_shape
        y = x.reshape(new_shape)
        return y
    

def discretize_functions(f_list, x):
    """
    Discretize a list of functions over a given input.

    Args:
        f_list (list): A list of callable functions.
        x (np.ndarray): Input points to evaluate the functions. Shape: (n_points,).

    Returns:
        np.ndarray: Discretized function values. Shape: (len(f_list), n_points).
    """
    f_discretized = Parallel(n_jobs=-1)(delayed(f)(x) for f in f_list)
    f_discretized = np.array(f_discretized)
    return f_discretized


def balance_num_trainable_params(model, N_w):
    """
    Adjust the number of trainable parameters in a model to match a target budget.
    This function iteratively adjusts the `free_parameter` attribute of the model
    to ensure the total number of trainable parameters matches the target `N_w`.
    'N_w' serves as an upper limit for the number of trainable parameters: 
    the model will thus have at most 'N_w' trainable parameters.
    The model's `init_layers` method is called to reinitialize the layers after each adjustment.

    Args:
        model (torch.nn.Module): The model to adjust.
        N_w (int): Target number of trainable parameters.

    Returns:
        torch.nn.Module: The adjusted model with the desired number of trainable parameters.
        
    Raises:
        ValueError: If the model cannot meet the trainable parameter budget.
    """
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