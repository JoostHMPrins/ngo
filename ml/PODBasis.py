import numpy as np
from BSplines import *

class BSplineInterpolatedPOD2D:
    def __init__(self, x_data, u_data, h, knots_x, knots_y, polynomial_order):
        self.basis_2d = BSplineBasis2D(knots_x, knots_y, polynomial_order)
        self.num_basis_1d = self.basis_2d.basis_1d_x.num_basis
        self.p = self.basis_2d.p
        # self.PODbasis = self.compute_PODbasis(u_data, h)
        # self.PODcoeffs = self.compute_PODcoeffs(x_data, u_data, h, knots_x, knots_y, polynomial_order)
        self.PODcoeffs = np.load('../../../trainingdata/PODcoeffs.npy')
        # self.errors = self.compute_projection_errors(x_data)
        
    def compute_PODbasis(self, u_data, h):
        U, self.singularvalues, Vstar = np.linalg.svd(u_data.T, full_matrices=False)
        PODbasis = U.T
        PODbasis_truncated = PODbasis[:h]
        return PODbasis_truncated
    
    def compute_PODcoeffs(self, x_data, u_data, h, knots_x, knots_y, polynomial_order):
        PODbasis = self.compute_PODbasis(u_data, h)
        PODcoeffs = []
        for i in range(len(PODbasis)):
            interpolator = BSplineInterpolator2D(x_data, PODbasis[i], knots_x, knots_y, polynomial_order)
            PODcoeffs.append(interpolator.compute_coeffs(x_data, PODbasis[i]))
        PODcoeffs = np.array(PODcoeffs)
        np.save('../../../PODcoeffs.npy', PODcoeffs)
        return PODcoeffs
        
    def forward(self, x):
        psi = self.basis_2d.forward(x)
        output = opt_einsum.contract('nm,Nm->Nn', self.PODcoeffs, psi)
        return output
    
    def grad(self, x):
        gradpsi = self.basis_2d.grad(x)
        output = opt_einsum.contract('nm,Nmi->Nni', self.PODcoeffs, gradpsi)
        return output
    
    def compute_projection_errors(self, x_data):
        PODbasis_int = self.forward(x_data)
        errors = np.linalg.norm(PODbasis_int.T - self.PODbasis, ord=2, axis=-1)/np.linalg.norm(self.PODbasis, ord=2, axis=-1)
        return errors