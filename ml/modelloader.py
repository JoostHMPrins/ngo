import os
import torch

def loadmodelfromlabel(model, label, logdir, sublogdir, map_location):
    
    for file in os.listdir(logdir + '/' + sublogdir + '/' + label):
        if file.endswith('.ckpt'):
            ckpt = torch.load(logdir + '/' + sublogdir + '/' + label + '/' + file,  map_location=map_location)
            hparams = ckpt['hyper_parameters']
            params = ckpt['params']
            Model = model(params)
            statedict = ckpt['state_dict']
            Model.load_state_dict(statedict)
            Model.hparams['device'] = map_location
    return Model