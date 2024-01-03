import numpy as np
import os
import json
import dill


def save_data(params, variable, data, savedir):
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    with open(savedir+'/params.json', 'w') as fp:
        json.dump(params, fp)
    np.save(savedir+'/'+variable+'.npy', data)
    
        
def save_function_list(params, variable, function_list, savedir):
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    with open(savedir+'/params.json', 'w') as fp:
        json.dump(params, fp)
    # Serialize the function list
    serialized_functions = dill.dumps(function_list)
    # Save the serialized functions to a file
    with open(savedir+'/'+variable+'.pkl', 'wb') as file:
        file.write(serialized_functions)
        
        
def load_function_list(variable, loaddir):
    with open(loaddir+'/'+variable+'.pkl', 'rb') as file:
        serialized_functions = file.read()
    deserialized_functions = dill.loads(serialized_functions)
    return deserialized_functions