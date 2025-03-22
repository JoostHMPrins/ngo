# Copyright 2025 Joost Prins

# Local
import os

# 3rd Party
import torch

def loadmodelfromlabel(model, label, logdir, sublogdir, device):
    
    for file in os.listdir(logdir + '/' + sublogdir + '/' + label):
        if file.endswith('.ckpt'):
            ckpt = torch.load(logdir + '/' + sublogdir + '/' + label + '/' + file,  map_location=device)
            hparams = ckpt['hparams']
            hparams['used_device'] = device
            Model = model(hparams)
            statedict = ckpt['state_dict']
            Model.load_state_dict(statedict)
    return Model