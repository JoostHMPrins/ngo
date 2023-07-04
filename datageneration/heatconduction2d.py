import numpy as np
import matplotlib.pyplot as plt
from nutils import mesh, function, solver
from nutils.expression_v2 import Namespace

from randompolynomials import randompoly1DO3, randompoly2DO3
from datasaver import savedata
    

def main(params, inputs, save, savedir, label):
    
    #Unit square geometry and mesh
    domain, geom = mesh.unitsquare(nelems=params['simparams']['nelems'], etype=params['simparams']['etype'])
    
    #Namespace
    ns = Namespace()
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.basis = domain.basis(params['simparams']['btype'], degree=params['simparams']['basisdegree'])
    ns.u = function.dotarg('lhs', ns.basis) #Solution
    
    theta = inputs['theta'] #Conductivity
    f = inputs['f'] #Forcing
    etat = inputs['etat'] #Neumann BC top
    etab = inputs['etab'] #Neumann BC bottom
    gl = inputs['gl'] #Dirichlet BC left
    gr = inputs['gr'] #Dirichlet BC right
    
    ns.theta = theta(ns.x[0], ns.x[1])
    ns.f = f(ns.x[0], ns.x[1])
    ns.etat = etat(ns.x[0])
    ns.etab = etab(ns.x[0])
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
    x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)
        
    outputs = {'x': x, 'u': u}
    
    return outputs


def postprocessdata(params, inputs, outputs):
    
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
    sensornodes_Omega = np.vstack([x_sensor.ravel(), y_sensor.ravel()]).T
    theta_sensor = theta(sensornodes_Omega[:,0], sensornodes_Omega[:,1])
    f_sensor  = f(sensornodes_Omega[:,0], sensornodes_Omega[:,1])

    #Sensor nodes grid boundary, sampling of eta at sensor nodes
    sensornodes_Gamma = np.linspace(0,1,int(N_sensornodes/2))
    etab_sensor = etab(sensornodes_Gamma)
    etat_sensor = etat(sensornodes_Gamma)
    eta_sensor = np.concatenate((etab_sensor, etat_sensor))
    
    #Sampling of x and u at random output points
    indices = np.linspace(0,x.shape[0]-1, x.shape[0], dtype=int)
    indices_output = np.random.choice(indices, size=N_outputnodes, replace=False)
    x_output = x[indices_output]
    u_output = u[indices_output]
    
    #Stacking theta, f and eta N_outputnodes times, to get N_outputnodes (theta,f,eta,x,u) training data samples
    theta_post = np.tile(theta_sensor, (N_outputnodes, 1))
    f_post = np.tile(f_sensor, (N_outputnodes, 1))
    eta_post = np.tile(eta_sensor, (N_outputnodes, 1))
    x_post = x_output
    u_post = u_output
    
    data_postprocessed = {}
    data_postprocessed['theta'] = theta_post
    data_postprocessed['f'] = f_post
    data_postprocessed['eta'] = eta_post
    data_postprocessed['x'] = x_post
    data_postprocessed['u'] = u_post
    
    return data_postprocessed 


def datasetgenerator(params, save, savedir, label):
    
    theta_array = []
    f_array = []
    eta_array = []
    x_array = []
    u_array = []
    
    for i in range(params['trainingdataparams']['N_sims']):
        
        print("Simulation: "+str(i))
        
        #Generate random input data
        c_theta = np.random.uniform(-0.1, 0.1, 10)
        c_f = np.random.uniform(-0.1, 0.1, 10)
        c_etab = np.random.uniform(-0.1, 0.1, 4)
        c_etat = np.random.uniform(-0.1, 0.1, 4)
        theta = randompoly2DO3(c_theta)
        f = randompoly2DO3(c_f)
        etab = randompoly1DO3(c_etab)
        etat = randompoly1DO3(c_etat)
        gl = 0
        gr = 0

        #Perform simulations
        inputs = {'theta': theta, 'f': f, 'etab': etab, 'etat': etat, 'gl': gl, 'gr': gr}
        outputs = main(params, inputs, save=False, savedir='.', label='.')
        
        #Postprocess data
        data_postprocessed = postprocessdata(params, inputs, outputs)
        
        #Collect data
        theta_array.append(data_postprocessed['theta'])
        f_array.append(data_postprocessed['f'])
        eta_array.append(data_postprocessed['eta'])
        x_array.append(data_postprocessed['x'])
        u_array.append(data_postprocessed['u'])
    
    #Convert to numpy and save
    theta_array = np.array(theta_array)
    f_array = np.array(f_array)
    eta_array = np.array(eta_array)
    x_array = np.array(x_array)
    u_array = np.array(u_array)
    data = {'theta': theta_array, 'f': f_array, 'eta': eta_array, 'x': x_array, 'u': u_array}
    if save==True:
        savedata(params, data, savedir, label)