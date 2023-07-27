import numpy as np
from scipy.interpolate import Rbf
from numba import jit
import typing
from nutils import function, evaluable
from nutils.sparse import dtype, toarray


# @jit
def GRF2D(N_gridpoints, l, positive):

    #Compute covariance matrix of points on a grid
    X, Y = np.mgrid[0:1:N_gridpoints*1j, 0:1:N_gridpoints*1j]
    x = np.vstack([X.ravel(), Y.ravel()]).T
    cov = np.zeros((N_gridpoints**2,N_gridpoints**2))
    for i in range(N_gridpoints**2):
        for j in range(N_gridpoints**2):
            cov[i,j] = np.exp(-np.sum((x[i] - x[j])**2, axis=-1)/(2*l**2))
    
    #Compute GRF on grid
    GRF = np.random.multivariate_normal(np.zeros(N_gridpoints**2), cov=cov, size=1)

    #Scale data and make positive if desired
    DeltaGRF = np.amax(GRF) - np.amin(GRF)
    GRF = GRF/DeltaGRF*0.97
    if positive==True:    
        GRF = GRF - np.amin(GRF) + 0.02

    #Interpolate GRF data with RBF interpolator
    GRF = GRF.flatten()
    GRFfunction = Rbf(x[:,0], x[:,1], GRF, function='gaussian', epsilon=l)
    
    def func(x_0,x_1):
        output = GRFfunction(x_0,x_1)
        return output
    
    return func
 

class ArcTan(evaluable.Pointwise):
    'Inverse tangent, element-wise.'
    evalf = staticmethod(GRF2D)
    complex_deriv = lambda x: None,
    return_type = float
    

IntoArray = typing.Union['Array', np.ndarray, bool, int, float, complex]

def grf2d(*__arg: IntoArray) -> function.Array:
    '''Return the trigonometric inverse tangent of the argument, elementwise.

    Parameters
    ----------
    arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

    Returns
    -------
    :class:`Array`
    '''

    return function._Wrapper.broadcasted_arrays(ArcTan, __arg, min_dtype=float)


# @jit
def GRF1D(N_gridpoints, l):

    #Compute covariance matrix of points on a grid
    x = np.linspace(0,1,N_gridpoints)
    cov = np.zeros((N_gridpoints,N_gridpoints))
    for i in range(N_gridpoints):
        for j in range(N_gridpoints):
            cov[i,j] = np.exp(-np.sum((x[i] - x[j])**2, axis=-1)/(2*l**2))
    
    #Compute GRF on grid
    GRF = np.random.multivariate_normal(np.zeros(N_gridpoints), cov=cov, size=1)

    #Scale data to [-1,1]
    DeltaGRF = np.amax(GRF) - np.amin(GRF)
    GRF = GRF/DeltaGRF*2
    GRF = GRF - np.amin(GRF) - 1 

    #Interpolate GRF data with RBF interpolator
    GRF = GRF.flatten()
    GRFfunction = Rbf(x, GRF, function='gaussian', epsilon=l)
    
    return GRFfunction