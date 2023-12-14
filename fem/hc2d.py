import numpy as np
import matplotlib.pyplot as plt
from nutils import mesh, function, solver
from nutils.expression_v2 import Namespace


#*args = save, savedir, label
#**kwargs = **simparams, **trainingdataparams, **inputs
def main(*args, **kwargs):
    
    #Unit square geometry and mesh
    domain, geom = mesh.unitsquare(nelems=kwargs['nelems'], etype=kwargs['etype'])
    
    #Namespace
    ns = Namespace()
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.basis = domain.basis(kwargs['btype'], degree=kwargs['basisdegree'])
    ns.u = function.dotarg('lhs', ns.basis) #Solution
    
    #Inputs
    theta = kwargs['theta'] #Conductivity
    f = kwargs['f'] #Forcing
    etat = kwargs['etat'] #Neumann BC top
    etab = kwargs['etab'] #Neumann BC bottom
    gl = kwargs['gl'] #Dirichlet BC left
    gr = kwargs['gr'] #Dirichlet BC right
    ns.theta = theta(ns.x)
    ns.f = f(ns.x)
    ns.etat = etat(ns.x)
    ns.etab = etab(ns.x)
    ns.gl = gl
    ns.gr = gr
        
    #Residual
    res = domain.integral('∇_i(basis_n) theta ∇_i(u) dV' @ ns, degree=kwargs['intdegree']) #Stiffness
    res -= domain.integral('basis_n f dV' @ ns, degree=kwargs['intdegree']) #Forcing
    res -= domain.boundary['top'].integral('basis_n etat dS' @ ns, degree=kwargs['intdegree']) #Neumann BC
    res -= domain.boundary['bottom'].integral('basis_n etab dS' @ ns, degree=kwargs['intdegree']) #Neumann BC
    
    #Dirichlet BC
    sqr = domain.boundary['left'].integral('(u - gl)^2 dS' @ ns, degree=kwargs['intdegree'])
    sqr += domain.boundary['right'].integral('(u - gr)^2 dS' @ ns, degree=kwargs['intdegree'])
    cons = solver.optimize('lhs', sqr, droptol=1e-15)

    #Solve system
    lhs = solver.solve_linear('lhs', res, constrain=cons)
    
    #Sampling of the input functions and solution
    bezier = domain.sample('bezier', kwargs['nfemsamples'])
    x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)
        
    return x, u