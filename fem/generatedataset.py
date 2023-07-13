from heatconduction2d import datasetgenerator

inputdataparams = {'type': 'polynomial',
                   'N_grfpoints': None,
                   'l_theta': None,
                   'positive_theta': None,
                   'l_f': None,
                   'positive_f': None,
                   'l_eta': None,
                   'positive_eta': None}
                   

simparams = {'nelems': 32,
             'etype': 'square',
             'btype': 'spline',
             'basisdegree': 1,
             'intdegree': 2,
             'nfemsamples': 2}

trainingdataparams = {'N_sensornodes': 144,
                      'N_outputnodes': 268,
                      'N_samples': 10000}

params = {}
params['inputdataparams'] = inputdataparams
params['simparams'] = simparams
params['trainingdataparams'] = trainingdataparams

datasetgenerator(params, save=True, savedir='../../../trainingdata', label='polynomial_large')