import numpy as np
import os
import json

def savedata(params, data, savedir, label):
    
    if not os.path.isdir(savedir+'/'+label):
        os.makedirs(savedir+'/'+label)
        
    with open(savedir+'/'+label+'/params.json', 'w') as fp:
        json.dump(params, fp)
    
    for key, value in data.items() :
        np.save(savedir+'/'+label+'/'+key+'.npy', value)