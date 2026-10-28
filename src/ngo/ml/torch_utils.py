# Copyright (c) 2023-2025 Eindhoven University of Technology

# 3rd Party
import torch

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