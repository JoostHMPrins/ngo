# Copyright 2025 Joost Prins

# Local
import os

# 3rd Party
import torch

def loadmodelfromlabel(model, label, logdir, sublogdir, device):
    """
    Load a PyTorch model from a checkpoint file based on a given label.

    This function searches for a checkpoint file in the specified logging directory
    and subdirectory, loads the model's state dictionary, and initializes the model
    with the corresponding hyperparameters.

    Args:
        model (class): The model class to be instantiated.
        label (str): A label identifying the experiment version.
        logdir (str): The base directory for logging.
        sublogdir (str): The subdirectory for the specific experiment.
        device (str or torch.device): The device to map the model to (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded model with its state dictionary and hyperparameters.
    """
    for file in os.listdir(logdir + '/' + sublogdir + '/' + label):
        if file.endswith('.ckpt'):
            ckpt = torch.load(logdir + '/' + sublogdir + '/' + label + '/' + file, weights_only=False,  map_location=device)
            hparams = ckpt['hparams']
            hparams['used_device'] = device
            Model = model(hparams)
            statedict = ckpt['state_dict']
            Model.load_state_dict(statedict)
    return Model