import numpy as np
import matplotlib.pyplot as plt
from nutils import mesh, function, solver
from nutils.expression_v2 import Namespace

from randompolynomials import randompoly1DO3, randompoly2DO3
from datasaver import savedata
    

def main(params, inputs, save, savedir, label):
    
    #Unit square geometry and mesh
    domain, geom = mesh.unitsquare(nelems=params['nelems'], etype=params['etype'])
    
    #Namespace
    ns = Namespace()
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.basis = domain.basis(params['btype'], degree=params['basisdegree'])
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
    res = domain.integral('∇_i(basis_n) theta ∇_i(u) dV' @ ns, degree=params['intdegree']) #Stiffness
    res -= domain.integral('basis_n f dV' @ ns, degree=params['intdegree']) #Forcing
    res -= domain.boundary['top'].integral('basis_n etat dS' @ ns, degree=params['intdegree']) #Neumann BC
    res -= domain.boundary['bottom'].integral('basis_n etab dS' @ ns, degree=params['intdegree']) #Neumann BC
    
    #Dirichlet BC
    sqr = domain.boundary['left'].integral('(u - gl)^2 dS' @ ns, degree=params['intdegree'])
    sqr += domain.boundary['right'].integral('(u - gr)^2 dS' @ ns, degree=params['intdegree'])
    cons = solver.optimize('lhs', sqr, droptol=1e-15)

    # residual vector evaluates to zero in the corresponding entries. This step
    # involves a linearization of ``res``, resulting in a jacobian matrix and
    # right hand side vector that are subsequently assembled and solved. The
    # resulting ``lhs`` array matches ``cons`` in the constrained entries.
    lhs = solver.solve_linear('lhs', res, constrain=cons)
    
    #Sampling of the input functions and solution
    bezier = domain.sample('bezier', params['nfemsamples'])
    x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)
        
    outputs = {'x': x,
               'u': u}
    
    return outputs