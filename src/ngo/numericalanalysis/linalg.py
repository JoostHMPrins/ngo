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


def kron_innerproduct(x, A_i, y):
    """Compute the inner product <x, (A_0 ⊗ A_1 ⊗ ... ⊗ A_{d-1}) y>
    for a batch of Kronecker-product matrices and vectors without forming
    the full Kronecker product.
    Args:
        x: Tensor of shape (N_samples, n_0, n_1, ..., n_{d-1})
        A_i: Tuple of d Tensors, each of shape (N_samples, n_i, n_i)
        y: Tensor of shape (N_samples, n_0, n_1, ..., n_{d-1})
    Returns:
        inner_products: Tensor of shape (N_samples,)
    """
    # Compute Ay = (A_0 ⊗ A_1 ⊗ ... ⊗ A_{d-1}) y
    Ay = kron_matvec(A_i, y)
    # Compute inner products <x, Ay>
    xAy = opt_einsum.contract('bi...,bi...->b', x, Ay)
    return xAy


def kron_solve(A_i, b, left=False):
    """Solve a (batched) Kronecker-product system.
    Computes the equivalent of x = (A_0 ⊗ A_1 ⊗ ... ⊗ A_{d-1})^{-1} @ b without
    forming the full Kronecker product. The operation is performed by a
    sequence of solves and supports an optional leading batch
    dimension.

    Inputs:
    A_i: Sequence of d factors, where each A_i has shape 
         (n_i, n_i) for i in [0, d-1].
    b: Right-hand side vector reshaped into Kronecker product dimensions as
       (B, N_0, N_1, ..., N_{d-1}) 
    left: Boolean indicating whether to solve left or right  systems.

    Returns:
    x: Solution vector with shape (B, N_0, N_1, ..., N_{d-1}).

    Notes:
    - The Kronecker factors A_i do not have a leading batch dimension,
    while the vector b does have a leading batch dimension.
    """
    # Initialize x as b
    x = b
    # Sequentially apply each Kronecker factor
    for i in range(len(A_i)):
        # print(i)
        # # Move the current dimension to the last position
        x = x.moveaxis(1,-1)
        # Solve the system for the current dimension
        x = torch.linalg.solve(A_i[i], x, left=left)
    return x

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


def kron_inv(A_i):
    """
    Compute the Kronecker product of the inverses of a d-tuple of matrices.

    Args:
        A_i (tuple of torch.Tensor): A tuple containing d matrices. Each matrix should be a 2D tensor of shape (m_i, n_i).
    Returns:
        torch.Tensor: The inverses of the Kronecker factors, resulting in a d-tuple of 2D tensors of shape (n_i, m_i).
    """
    A_inv = ()
    for i in range(len(A_i)):
        A_inv += (torch.linalg.inv(A_i[i]),)
    return A_inv


def integrate(W_q, f_q):
    """
    Integrate a function represented at quadrature points using given quadrature weights.

    Args:
        W_q (tuple of torch.Tensor): A tuple containing d tensors of quadrature weights. Each tensor should have shape (Q_i,).
        f_q (torch.Tensor): A tensor representing the function values at quadrature points. Shape: (B, Q_0, Q_1, ..., Q_{d-1}).

    Returns:
        torch.Tensor: The integrated values. Shape: (B,).
    """
    integral = f_q
    for i in range(len(W_q)):
        integral = opt_einsum.contract('q,Bq...->B...', W_q[i], integral)
    return integral


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
            Theta_i = theta.basis.bases[i].forward(X_q[i]) #shape (B,n,q)
        elif operation == 1:
            Theta_i = theta.basis.bases[i].grad(X_q[i]) #shape (B,n,q)
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


def kron_vecT(U_m, T_i):
    """
    Assemble the Kronecker double product factors A_i by contracting a set of 
    Kronecker triple products T_i with coefficients U_m.

    Args:
        U_m: Coefficient tensor of shape (B,l_0,l_1,...,l_{d-1})
        T_i: d tuple of Kronecker triple products, each of shape (B,l_i,m_i,n_i)
        
    Returns:
        A: d tuple of resulting Kronecker factors, each of shape (B,m_i,n_i)
    """
    # Initialize empty A tuple
    A = ()
    # Sequentially contract each coefficient tensor axis with each Kronecker triple product
    for i in range(len(T_i)):
        A_i = opt_einsum.contract('Bl...,Blmn->Bmn', U_m, T_i[i])
        A += (A_i,)
        # Move the new contraction axis to position 1
        U_m = U_m.moveaxis(1, -1)
    return A


def kron_single(F_i):
    """
    Compute the Kronecker product of a d-tuple of matrices.

    Args:
        F_i (tuple of torch.Tensor): A tuple containing d matrices. Each matrix should be a 2D tensor of shape (m_i, n_i).

    Returns:
        torch.Tensor: The Kronecker product of the input matrices, resulting in a 2D tensor of shape (m_0*m_1*...*m_{d-1}, n_0*n_1*...*n_{d-1}).
    """
    B = F_i[0]
    for i in range(1, len(F_i)):
        B = torch.kron(B, F_i[i])
    return B


def kron(F_i):
    """
    Compute the Kronecker product for a batch of d-tuples of matrices.

    Args:
        F_i (tuple): A tuple where each element is a batch of matrices of shape (B, m_i, n_i).

    Returns:
        torch.Tensor: A batch of Kronecker products, each of shape (m_0*m_1*...*m_{d-1}, n_0*n_1*...*n_{d-1}).
    """
    return torch.vmap(kron_single)(F_i)


def add_tuples(t1, t2):
    """Return element-wise sum of two tuples."""
    return tuple(a + b for a, b in zip(t1, t2))


def subtract_tuples(t1, t2):
    """Return element-wise difference of two tuples (t1 - t2)."""
    return tuple(a - b for a, b in zip(t1, t2))

