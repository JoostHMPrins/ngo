import numpy as np
import os
import json

def savedata(params, data, arraynames, savedir, label):
    
    if not os.path.isdir(savedir+'/'+label):
        os.makedirs(savedir+'/'+label)
        
    with open(savedir+'/'+label+'/params.json', 'w') as fp:
        json.dump(params, fp)
    
    for i in range(len(data)):
        np.save(savedir+'/'+label+'/'+arraynames[i], data[i])