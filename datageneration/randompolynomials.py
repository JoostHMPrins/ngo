import numpy as np


def randompoly1DO3(x):
    c = np.random.uniform(-1,1,4)
    poly = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3
    return poly


def randompoly2DO3(x,y):
    c = np.random.uniform(-1,1,10)
    poly = c[0] + c[1]*x + c[2]*y + c[3]*x**2 + c[4]*y**2 + c[5]*x*y + c[6]*x**3 + c[7]*y**3 + c[8]*x*y**2 + c[9]*y*x**2
    return poly


def main(nelems, etype, btype, degree, inputdata, nsamples):

    #Unit square geometry and mesh
    domain, geom = mesh.unitsquare(nelems, etype)
    
    #Namespace
    ns = Namespace()
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.basis = domain.basis(btype, degree=degree)
    ns.u = function.dotarg('lhs', ns.basis) #Solution
    
    if inputdata=='poly':
        theta = randompoly2DO3 #Conductivity
        f = randompoly2DO3 #Forcing
        etat = randompoly1DO3 #Neumann BC top
        etab = randompoly1DO3 #Neumann BC bottom
    gl = 0 #Dirichlet BC left
    gr = 0 #Dirichlet BC right
    
    ns.theta = theta(ns.x[0], ns.x[1])
    ns.f = f(ns.x[0], ns.x[1])
    ns.etat = etat(ns.x[0])
    ns.etab = etab(ns.x[0])
    ns.gl = gl
    ns.gr = gr

    #Residual
    res = domain.integral('∇_i(basis_n) theta ∇_i(u) dV' @ ns, degree=degree*2) #Stiffness
    res -= domain.integral('basis_n f dV' @ ns, degree=degree*2) #Forcing
    res -= domain.boundary['top'].integral('basis_n etat dS' @ ns, degree=degree*2) #Neumann BC
    res -= domain.boundary['bottom'].integral('basis_n etab dS' @ ns, degree=degree*2) #Neumann BC
    
    #Dirichlet BC
    sqr = domain.boundary['left'].integral('(u - gl)^2 dS' @ ns, degree=degree*2)
    sqr += domain.boundary['right'].integral('(u - gr)^2 dS' @ ns, degree=degree*2)
    cons = solver.optimize('lhs', sqr, droptol=1e-15)

    # The unconstrained entries of ``?lhs`` are to be determined such that the
    # residual vector evaluates to zero in the corresponding entries. This step
    # involves a linearization of ``res``, resulting in a jacobian matrix and
    # right hand side vector that are subsequently assembled and solved. The
    # resulting ``lhs`` array matches ``cons`` in the constrained entries.
    lhs = solver.solve_linear('lhs', res, constrain=cons)
    
    #Sampling of the input functions and solution
    bezier = domain.sample('bezier', nsamples)
    x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)
    x, theta, f, etat, etab, gl, gr, u = bezier.eval(['x_i', 'theta', 'f', 'etat', 'etab', 'gl', 'gr', 'u'] @ ns, lhs=lhs)
    
    #Plots
#     vmin = np.amin([theta,f,u])
#     vmax = np.amax([theta,f,u])

#     plt.title('theta')
#     plt.tripcolor(x[:,0], x[:,1], theta)#, vmin=vmin, vmax=vmax)
#     plt.xlim(0,1)
#     plt.ylim(0,1)
#     plt.colorbar()
#     plt.show()
    
#     plt.title('f')
#     plt.tripcolor(x[:,0], x[:,1], f)#, vmin=vmin, vmax=vmax)
#     plt.xlim(0,1)
#     plt.ylim(0,1)
#     plt.colorbar()
#     plt.show()
    
#     plt.plot(x[:,0],etat, label='etat')
#     plt.plot(x[:,0],etab, label='etab')
#     plt.legend()
#     plt.show()

#     plt.title('u')
#     plt.tripcolor(x[:,0], x[:,1], u)#, vmin=vmin, vmax=vmax)
#     plt.xlim(0,1)
#     plt.ylim(0,1)
#     plt.colorbar()
#     plt.show()
    
    return x, theta, f, etat, etab, gl, gr, u