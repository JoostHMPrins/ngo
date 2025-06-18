import os
import torch

def loadmodelfromlabel(model, label, logdir, sublogdir, device):
    
    for file in os.listdir(logdir + '/' + sublogdir + '/' + label):
        if file.endswith('.ckpt'):
            ckpt = torch.load(logdir + '/' + sublogdir + '/' + label + '/' + file,  map_location=device)
            hparams = ckpt['hparams']
            hparams['logdir'] = logdir + '/' + sublogdir + '/' + label
            hparams['used_device'] = device
            Model = model(hparams)
            statedict = ckpt['state_dict']
            Model = Model.to(hparams['dtype'])
            Model.load_state_dict(statedict)
    return Model