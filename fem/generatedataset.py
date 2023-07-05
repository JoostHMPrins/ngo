from heatconduction2d import datasetgenerator

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
params['simparams'] = simparams
params['trainingdataparams'] = trainingdataparams

datasetgenerator(params, save=True, savedir='../../../trainingdata', label='test2')