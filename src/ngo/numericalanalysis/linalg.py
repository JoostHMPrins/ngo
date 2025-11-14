# Copyright 2025 Joost Prins

# 3rd Party
import torch

def kron_matvec(A_i, x):
    """Apply product of a (batched) Kronecker-product matrix and a (batched) vector.
    Computes the equivalent of y = (A_0 ⊗ A_1 ⊗ ... ⊗ A_{d-1}) @ x without
    forming the full Kronecker product. The operation is performed by a
    sequence of tensor contractions and supports an optional leading batch
    dimension.

    Inputs:
    A_i: Sequence of d factors, where each A_i has shape 
         (B, n_i, n_i) for i in [0, d-1].
    x: Vector reshaped into Kronecker product dimensions as(B, n_0, n_1, ..., n_{d-1}) 
       
    Returns:
    b: Result of applying the Kronecker-product matrix to x, with shape
       (B, n_0, n_1, ..., n_{d-1}).

    Notes:
    - Implementation uses einsum to rely on per-batch operations. 
      An torch.vmap + torch.tensordot variant is kept commented for reference.
    """
    # Initialize b as x
    b = x
    # Sequentially apply each Kronecker factor
    for i in range(len(A_i)):
        # b = torch.vmap(torch.tensordot)(b, A_i[i], dims=([0], [1]))
        b = torch.einsum('Nj...,Nkj->N...k', b, A_i[i])
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
       (B, n_0, n_1, ..., n_{d-1}) 
    upper: Boolean indicating whether the factors L_i are upper or lower
           triangular.
    left: Boolean indicating whether to solve left or right triangular systems.

    Returns:
    x: Solution vector with shape (B, n_0, n_1, ..., n_{d-1}).

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


def torch_repeat(x, repeats, axis=None):
    """
    PyTorch equivalent of np.repeat.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    repeats : int or 1D list/Tensor of ints
        Number of repetitions.
    axis : int or None
        Dimension along which to repeat. If None, the tensor is flattened.

    Returns
    -------
    torch.Tensor
    """

    # Match NumPy: if axis is None, flatten first
    if axis is None:
        x = x.flatten()
        return torch.repeat_interleave(x, repeats)

    # Normalize negative axes
    if axis < 0:
        axis += x.dim()

    # Move target axis to the end
    x = x.transpose(axis, -1)

    # Apply repeat_interleave on last dimension
    out = torch.repeat_interleave(x, repeats, dim=-1)

    # Move axis back
    out = out.transpose(axis, -1)

    return out