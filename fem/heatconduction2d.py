import numpy as np
import matplotlib.pyplot as plt
from nutils import mesh, function, solver
from nutils.expression_v2 import Namespace

from randompolynomials import randompoly1DO3, randompoly2DO3, randompoly2DO3sqr
from GRF import GRF 
from datasaver import savedata

def main(params, inputs, sample, save, savedir, label):
    
    #Unit square geometry and mesh
    domain, geom = mesh.unitsquare(nelems=params['simparams']['nelems'], etype=params['simparams']['etype'])
    
    #Namespace
    ns = Namespace()
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.basis = domain.basis(params['simparams']['btype'], degree=params['simparams']['basisdegree'])
    ns.u = function.dotarg('lhs', ns.basis) #Solution
    
    if params['trainingdataparams']['inputdata']=='grf':
        theta = inputs['theta'].sample(inputs['theta'].RBFint_pointwise_scaled, sample) #Conductivity
        f = inputs['f'].sample(inputs['f'].RBFint_pointwise_scaled, sample) #Forcing
        etat = inputs['etat'].sample(inputs['etat'].RBFint_pointwise_scaled, sample) #Neumann BC top
        etab = inputs['etab'].sample(inputs['etab'].RBFint_pointwise_scaled, sample) #Neumann BC bottom
        gl = inputs['gl'] #Dirichlet BC left
        gr = inputs['gr'] #Dirichlet BC right
    
    # ns.theta = theta(ns.x[0], ns.x[1])
    # ns.f = f(ns.x[0], ns.x[1])
    # ns.etat = etat(ns.x[0])
    # ns.etab = etab(ns.x[0])
    ns.theta = theta(ns.x)
    ns.f = f(ns.x)
    ns.etat = etat(ns.x)
    ns.etab = etab(ns.x)
    ns.gl = gl
    ns.gr = gr

    #Residual
    res = domain.integral('∇_i(basis_n) theta ∇_i(u) dV' @ ns, degree=params['simparams']['intdegree']) #Stiffness
    res -= domain.integral('basis_n f dV' @ ns, degree=params['simparams']['intdegree']) #Forcing
    res -= domain.boundary['top'].integral('basis_n etat dS' @ ns, degree=params['simparams']['intdegree']) #Neumann BC
    res -= domain.boundary['bottom'].integral('basis_n etab dS' @ ns, degree=params['simparams']['intdegree']) #Neumann BC
    
    #Dirichlet BC
    sqr = domain.boundary['left'].integral('(u - gl)^2 dS' @ ns, degree=params['simparams']['intdegree'])
    sqr += domain.boundary['right'].integral('(u - gr)^2 dS' @ ns, degree=params['simparams']['intdegree'])
    cons = solver.optimize('lhs', sqr, droptol=1e-15)

    # residual vector evaluates to zero in the corresponding entries. This step
    # involves a linearization of ``res``, resulting in a jacobian matrix and
    # right hand side vector that are subsequently assembled and solved. The
    # resulting ``lhs`` array matches ``cons`` in the constrained entries.
    lhs = solver.solve_linear('lhs', res, constrain=cons)
    
    #Sampling of the input functions and solution
    bezier = domain.sample('bezier', params['simparams']['nfemsamples'])
    x, theta, f, etab, etat, u = bezier.eval(['x_i', 'theta', 'f', 'etab', 'etat', 'u'] @ ns, lhs=lhs)
        
    outputs = {'x':x, 'theta':theta, 'f':f, 'etab':etab, 'etat':etat, 'u':u}
    
    return outputs


def postprocessdata(params, inputs, sample, outputs):
    
    N_sensornodes = params['trainingdataparams']['N_sensornodes']
    N_outputnodes = params['trainingdataparams']['N_outputnodes']
    theta = inputs['theta']
    f = inputs['f']
    etab = inputs['etab']
    etat = inputs['etat']
    x = outputs['x']
    u = outputs['u']

    #Sensor nodes grid for domain, sampling of theta and f at sensor nodes
    x_sensor, y_sensor = np.mgrid[0:1:np.sqrt(N_sensornodes)*1j, 0:1:np.sqrt(N_sensornodes)*1j]
    sensornodes = np.vstack([x_sensor.ravel(), y_sensor.ravel()]).T
    
    #Sensor data
    theta_sensor = theta.sample(theta.RBFint_scaled, sample)(sensornodes).reshape(12,12)
    f_sensor = f.sample(f.RBFint_scaled, sample)(sensornodes).reshape(12,12)
    etab_sensor = etab.sample(etab.RBFint_scaled, sample)(sensornodes).reshape(12,12)
    etat_sensor = etat.sample(etat.RBFint_scaled, sample)(sensornodes).reshape(12,12)
    #indicators of boundaries
    Gamma_etab = np.zeros(etab_sensor.shape)
    Gamma_etab[y_sensor==0] = 1
    Gamma_etat = np.zeros(etat_sensor.shape)
    Gamma_etat[y_sensor==1] = 1
    #set eta zero on non-boundary sites
    etab_sensor = Gamma_etab*etab_sensor
    etat_sensor = Gamma_etat*etat_sensor
    eta_sensor = etab_sensor + etat_sensor
    
    #Sampling of x and u at random output nodes
    indices = np.linspace(0,x.shape[0]-1, x.shape[0], dtype=int)
    indices_output = np.random.choice(indices, size=N_outputnodes, replace=False)
    x_output = x[indices_output]
    u_output = u[indices_output]
    
    #Put data in dict
    data_postprocessed = {}
    data_postprocessed['Theta'] = theta_sensor
    data_postprocessed['F'] = f_sensor
    data_postprocessed['N'] = eta_sensor
    data_postprocessed['x'] = x_output
    data_postprocessed['u'] = u_output
    
    return data_postprocessed 


def datasetgenerator(params, save, savedir, label):
    
    simparams = params['simparams']
    trainingdataparams = params['trainingdataparams']
    
    Theta_array = []
    F_array = []
    N_array = []
    x_array = []
    u_array = []
    
    for i in range(params['trainingdataparams']['N_samples']):
        
        print("Simulation: "+str(i))
        
        if params['trainingdataparams']['inputdata']=='polynomial':
            C = 0.2
            c_theta = C*np.random.uniform(-1, 1, 10)
            c_f = C*np.random.uniform(-1, 1, 10)
            c_etab = C*np.random.uniform(-1, 1, 4)
            c_etat = C*np.random.uniform(-1, 1, 4)
            theta = randompoly2DO3sqr(c_theta)
            f = randompoly2DO3sqr(c_f)
            etab = randompoly1DO3(c_etab)
            etat = randompoly1DO3(c_etat)
            gl = 0
            gr = 0
            
        if params['trainingdataparams']['inputdata']=='grf':
            theta = GRF(**simparams, **trainingdataparams, **trainingdataparams['theta'])
            f = GRF(**simparams, **trainingdataparams, **trainingdataparams['f'])
            etab = GRF(**simparams, **trainingdataparams, **trainingdataparams['eta'])
            etat = GRF(**simparams, **trainingdataparams, **trainingdataparams['eta'])
            gl = 0
            gr = 0

        #Perform simulations
        inputs = {'theta': theta, 'f': f, 'etab': etab, 'etat': etat, 'gl': gl, 'gr': gr}
        outputs = main(params, inputs, sample=i, save=False, savedir='.', label='.')
        
        #Postprocess data
        data_postprocessed = postprocessdata(params=params, inputs=inputs, sample=i, outputs=outputs)
        
        #Collect data
        Theta_array.append(data_postprocessed['Theta'])
        F_array.append(data_postprocessed['F'])
        N_array.append(data_postprocessed['N'])
        x_array.append(data_postprocessed['x'])
        u_array.append(data_postprocessed['u'])
    
    #Convert to numpy and save
    Theta_array = np.array(Theta_array)
    F_array = np.array(F_array)
    N_array = np.array(N_array)
    x_array = np.array(x_array)
    u_array = np.array(u_array)
    data = {'Theta': Theta_array, 'F': F_array, 'N': N_array, 'x': x_array, 'u': u_array}
    if save==True:
        savedata(params, data, savedir, label)