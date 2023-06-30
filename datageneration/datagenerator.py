import numpy as np

from heatconduction2d import main
from datasaver import savedata

params = {'inputdata': 'poly',
             'nelems': 10,
             'etype': 'square',
             'btype': 'spline',
             'basisdegree': 1,
             'intdegree': 2,
             'nfemsamples': 2,
             'N_samples': 10000}

x_array = []
theta_array = []
f_array = []
etat_array = []
etab_array = []
gl_array = []
gr_array = []
u_array = []

for i in range(params['N_samples']):

    print(i)

    x, theta, f, etat, etab, gl, gr, u = main(params, save=False, savedir='../../../trainingdata', label='polynomialdata')
    
    x_array.append(x)
    theta_array.append(theta)
    f_array.append(f)
    etat_array.append(etat)
    etab_array.append(etab)
    gl_array.append(gl)
    gr_array.append(gr)
    u_array.append(u)
    
x_array = np.array(x_array)
theta_array = np.array(theta_array)
f_array = np.array(f_array)
etat_array = np.array(etat_array)
etab_array = np.array(etab_array)
gl_array = np.array(gl_array)
gr_array = np.array(gr_array)
u_array = np.array(u_array)

data = [x_array, theta_array, f_array, etat_array, etab_array, gl_array, gr_array, u_array]
arraynames = ['x.npy','theta.npy','f.npy','etat.npy','etab.npy','gl.npy','gr.npy','u.npy']
savedata(params, data, arraynames, savedir='../../../trainingdata', label='test')