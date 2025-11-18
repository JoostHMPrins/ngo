# Copyright 2025 Joost Prins

# 3rd Party
import torch
import opt_einsum

def kron_matvec(A_i, x):
    """Apply product of a (batched) Kronecker-product matrix and a (batched) vector.
    Computes the equivalent of y = (A_0 ⊗ A_1 ⊗ ... ⊗ A_{d-1}) @ x without
    forming the full Kronecker product. The operation is performed by a
    sequence of tensor contractions and supports an optional leading batch
    dimension.

    Inputs:
    A_i: Sequence of d factors, where each A_i has shape 
         (B, N_i, N_i) for i in [0, d-1].
    x: Vector reshaped into Kronecker product dimensions as(B, N_0, N_1, ..., N_{d-1}) 
       
    Returns:
    b: Result of applying the Kronecker-product matrix to x, with shape
       (B, N_0, N_1, ..., N_{d-1}).

    Notes:
    - Implementation uses opt_einsum to rely on per-batch operations. 
      An torch.vmap + torch.tensordot variant is kept commented for reference.
    """
    # Initialize b as x
    b = x
    # Sequentially apply each Kronecker factor
    for i in range(len(A_i)):
        # b = torch.vmap(torch.tensordot)(b, A_i[i], dims=([0], [1]))
        b = opt_einsum.contract('Bj...,Bkj->B...k', b, A_i[i])
    return b


def kron_solve_triangular(L_i, b, upper, left=False):
    """Solve a (batched) Kronecker-product triangular system.
    Computes the equivalent of x = (L_0 ⊗ L_1 ⊗ ... ⊗ L_{d-1})^{-1} @ b without
    forming the full Kronecker product. The operation is performed by a
    sequence of triangular solves and supports an optional leading batch
    dimension.

    Inputs:
    L_i: Sequence of d factors, where each L_i has shape 
         (n_i, n_i) for i in [0, d-1].
    b: Right-hand side vector reshaped into Kronecker product dimensions as
       (B, N_0, N_1, ..., N_{d-1}) 
    upper: Boolean indicating whether the factors L_i are upper or lower
           triangular.
    left: Boolean indicating whether to solve left or right triangular systems.

    Returns:
    x: Solution vector with shape (B, N_0, N_1, ..., N_{d-1}).

    Notes:
    - The Kronecker factors L_i do not have a leading batch dimension,
    while the vector b does have a leading batch dimension.
    """
    # Initialize x as b
    x = b
    # Sequentially apply each Kronecker factor
    for i in range(len(L_i)):
        # print(i)
        # # Move the current dimension to the last position
        x = x.moveaxis(1,-1)
        # Solve the triangular system for the current dimension
        x = torch.linalg.solve_triangular(L_i[i], x, upper=upper, left=left)
    return x


def assemble_L(psi, f, W_q, X_q, derivative_orders):
    """
    Assemble the Kronecker factors of the linear form
    L(v) = \int f(x) \psi(x) dx
    or its derivatives.
    
    Args:
        psi: TensorProductBasis
        f: ManufacturedSolution
        W_q: d tuple of quadrature weights for each dimension, each of shape (Q_i,)
        X_q: d tuple of quadrature points for each dimension of shape (Q_i,)
        derivative_orders: tuple containing the derivative orders for psi in each dimension.
                           For example, (0,1,0) means no derivative for psi in dim 0, first derivative in dim 1, no derivative in dim 2.
    Returns:
        L: Tensor of shape (B,n_1,n_2,...,n_d)  
    """
    f_q = f.forward(X_q) 
    L = f_q
    for i in range(len(X_q)):
        # psi
        # Set derivative order for psi
        operation = derivative_orders[i]
        if operation == 0:
            Psi_i = psi.bases[i].forward(X_q[i]) 
        elif operation == 1:
            Psi_i = psi.bases[i].grad(X_q[i])
        # Contract over dimension i
        L = opt_einsum.contract('q,Bnq,Bq...->Bn...', W_q[i], Psi_i, L)
        # Move contracted index to the end
        L = L.moveaxis(1,-1)
    return L


def assemble_B(psi, phi, W_q, X_q, derivative_orders):
    """
    Assemble the Kronecker factors of the bilinear form
    B(u,v) = \int \psi(x) \phi(x) dx
    or its derivatives.
    
    Args:
        psi: TensorProductBasis
        phi: TensorProductBasis
        W_q: d tuple of quadrature weights for each dimension, each of shape (Q_i,)
        X_q: d tuple of quadrature points for each dimension of shape (Q_i,)
        derivative_orders: tuple of two tuples, each containing the derivative orders for psi and phi in each dimension.
                           For example, ((0,1,0),(1,0,0)) means no derivative for psi in dim 0, first derivative in dim 1, no derivative in dim 2,
                           and first derivative for phi in dim 0, no derivative in dim 1, no derivative in dim 2.
    
    Returns:
        F: d tuple of Kronecker factors, each of shape (B,n_i,m_i)
    """
    F = ()
    for i in range(len(X_q)):
        # psi
        # Set derivative order for psi
        operation = derivative_orders[0][i]
        if operation == 0:
            Psi_i = psi.bases[i].forward(X_q[i]) #shape (B,n,q)
        elif operation == 1:
            Psi_i = psi.bases[i].grad(X_q[i]) #shape (B,n,q)
        # phi
        # Set derivative order for phi
        operation = derivative_orders[1][i]
        if operation == 0:
            Phi_i = phi.bases[i].forward(X_q[i]) #shape (B,n,q)
        elif operation == 1:
            Phi_i = phi.bases[i].grad(X_q[i]) #shape (B,n,q)
        # F_i = opt_einsum.contract('q,Bmq->Bmq', W_q[i], Psi_i) #shape (B,n,Q)
        # F_i = opt_einsum.contract('Bmq,Bnq->Bmn', F_i, Phi_i) #shape (B,n,m)
        # Assemble in one step
        F_i = opt_einsum.contract('q,Bmq,Bnq->Bmn', W_q[i], Psi_i, Phi_i) #shape (B,n,m)
        F += (F_i,)
    return F


def assemble_T(theta, psi, phi, W_q, X_q, derivative_orders):
    """
    Assemble the Kronecker factors of the trilinear form
    T(u,v,w) = \int \theta(x) \psi(x) \phi(x) dx
    or its derivatives.

    Args:
        theta: TensorProductBasis
        psi: TensorProductBasis
        phi: TensorProductBasis
        W_q: d tuple of quadrature weights for each dimension, each of shape (Q_i,)
        X_q: d tuple of quadrature points for each dimension of shape (Q_i,)
        derivative_orders: tuple of three tuples, each containing the derivative orders for theta, psi and phi in each dimension.
                           For example, ((0,1,0),(1,0,0),(0,0,1)) means no derivative for theta in dim 0, first derivative in dim 1, no derivative in dim 2,
                           first derivative for psi in dim 0, no derivative in dim 1, no derivative in dim 2,
                           and no derivative for phi in dim 0, no derivative in dim 1, first derivative in dim 2.
    
    Returns:
        F: d tuple of Kronecker factors, each of shape (B,l_i,m_i,n_i)
    """
    F = ()
    for i in range(len(X_q)):
        # theta
        # Set derivative order for theta
        operation = derivative_orders[0][i]
        if operation == 0:
            Theta_i = theta.bases[i].forward(X_q[i]) #shape (B,n,q)
        elif operation == 1:
            Theta_i = theta.bases[i].grad(X_q[i]) #shape (B,n,q)
        # psi
        # Set derivative order for psi
        operation = derivative_orders[1][i]
        if operation == 0:
            Psi_i = psi.bases[i].forward(X_q[i]) #shape (B,n,q)
        elif operation == 1:
            Psi_i = psi.bases[i].grad(X_q[i]) #shape (B,n,q)
        # phi
        # Set derivative order for phi
        operation = derivative_orders[2][i]
        if operation == 0:
            Phi_i = phi.bases[i].forward(X_q[i]) #shape (B,n,q)
        elif operation == 1:
            Phi_i = phi.bases[i].grad(X_q[i]) #shape (B,n,q)
        # F_i = opt_einsum.contract('q,Blq->Blq', W_q[i], Theta_i) #shape (B,n,Q)
        # F_i = opt_einsum.contract('Blq,Bmq->Blmq', F_i, Psi_i) #shape (B,n,m)
        # F_i = opt_einsum.contract('Blmq,Bnq->Blmn', F_i, Phi_i) #shape (B,n,m)
        # Assemble in one step
        F_i = opt_einsum.contract('q,Blq,Bmq,Bnq->Blmn', W_q[i], Theta_i, Psi_i, Phi_i) #shape (B,n,m)
        F += (F_i,)
    return F