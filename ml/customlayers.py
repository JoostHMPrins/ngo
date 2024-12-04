import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, padding=0, dilation=1):
        """
        Generalized N-dimensional convolution module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.
            padding (int or tuple, optional): Zero-padding added to all sides. Default: 0.
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
            bias (bool, optional): If True, adds a learnable bias to the output. Default: True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Ensure kernel_size, stride, padding, and dilation are tuples matching spatial dimensions
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride if isinstance(stride, tuple) else (stride,) * len(self.kernel_size)
        self.padding = padding if isinstance(padding, tuple) else (padding,) * len(self.kernel_size)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * len(self.kernel_size)

        # Initialize weights and bias
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Forward pass for the generalized N-dimensional convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, *spatial_dims).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, *spatial_dims_out).
        """
        batch_size, in_channels, *spatial_dims = x.shape
        ndim = len(spatial_dims)

        # Adjust padding, stride, and dilation to match the number of dimensions
        assert len(self.kernel_size) == ndim, "Kernel size dimensions must match input dimensions"

        # Pad the input tensor
        pad = [(p, p) for p in reversed(self.padding)]  # Reverse to get correct padding order
        pad = [item for sublist in pad for item in sublist]  # Flatten padding list
        x_padded = torch.nn.functional.pad(x, pad, mode='constant', value=0)

        # Compute output spatial dimensions
        out_dims = [
            (spatial_dims[i] + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) // self.stride[i] + 1
            for i in range(ndim)
        ]

        # Extract patches using unfold (this is done sequentially for each spatial dimension)
        patches = x_padded.unfold(2, self.kernel_size[0], self.stride[0])
        for i in range(1, ndim):
            patches = patches.unfold(2 + i, self.kernel_size[i], self.stride[i])

        # Reshape patches to prepare for convolution
        patches = patches.contiguous().view(batch_size, in_channels, -1, *out_dims)

        # Perform convolution via einsum
        weight_shape = self.weight.shape
        weight = self.weight.view(self.out_channels, self.in_channels, -1)
        out = torch.einsum("bci...,oci->bo...", patches, weight)

        # Add bias if applicable
        if self.bias is not None:
            out += self.bias.view(1, -1, *[1] * ndim)

        return out


def balance_num_trainable_params(model, N_w):
    model.free_parameter = 1
    N_w_real_0 = model.hparams['N_w_real']
    free_parameter_values_list = []
    N_w_real = N_w_real_0
    while N_w_real < N_w:
        model.init_layers()
        N_w_real = N_w_real_0 + sum(p.numel() for p in model.parameters())
        free_parameter_values_list.append(model.free_parameter)
        model.free_parameter +=1
    if len(free_parameter_values_list)>1:
        model.free_parameter = free_parameter_values_list[-2]
    else:
        model.free_parameter = free_parameter_values_list[0]
    model.init_layers()
    count = sum(p.numel() for p in model.parameters())
    model.hparams['N_w_real'] += count
    if model.hparams['N_w_real']>N_w:
        raise ValueError('Not enough trainable parameter budget.')
    return model  


def discretize_functions(f_list, x, device):
    f_discretized = torch.zeros((len(f_list),x.shape[0]), device=device)
    for i in range(len(f_list)):
        print(i)
        f_discretized[i] = f_list[i](x)
    return f_discretized


def sort_matrices(matrices):
    """
    Sort the rows and columns of each matrix in the batch by decreasing L2 norm.
    
    Args:
        matrices (torch.Tensor): A batch of n x n matrices of shape (batch_size, n, n)
    
    Returns:
        torch.Tensor: A batch of sorted n x n matrices of shape (batch_size, n, n)
        torch.Tensor: Row sorted indices of shape (batch_size, n)
        torch.Tensor: Column sorted indices of shape (batch_size, n)
    """
    batch_size, n, _ = matrices.shape
    # Compute the L2 norm for rows and columns
    row_norms = torch.norm(matrices, dim=2, p=np.inf)  # shape: (batch_size, n)
    col_norms = torch.norm(matrices, dim=1, p=np.inf)  # shape: (batch_size, n)
    # Sort indices by decreasing L2 norm
    row_sorted_indices = torch.argsort(row_norms, dim=1, descending=True)  # shape: (batch_size, n)
    col_sorted_indices = torch.argsort(col_norms, dim=1, descending=True)  # shape: (batch_size, n)
    # Create a batch index tensor
    batch_indices = torch.arange(batch_size)#.unsqueeze(1)  # shape: (batch_size, 1)
    n_indices = torch.arange(n)
    # Sort rows and columns using advanced indexing
    sorted_matrices = matrices[batch_indices[:,None,None], row_sorted_indices[:,:,None], col_sorted_indices[:,None,:]]  # shape: (batch_size, n, n)
    return sorted_matrices, row_sorted_indices, col_sorted_indices

def unsort_matrices(sorted_matrices, row_sorted_indices, col_sorted_indices):
    """
    Unsort the matrices back to their original order.
    
    Args:
        sorted_matrices (torch.Tensor): A batch of sorted n x n matrices of shape (batch_size, n, n)
        row_sorted_indices (torch.Tensor): Row sorted indices of shape (batch_size, n)
        col_sorted_indices (torch.Tensor): Column sorted indices of shape (batch_size, n)
    
    Returns:
        torch.Tensor: A batch of unsorted n x n matrices of shape (batch_size, n, n)
    """
    batch_size, n, _ = sorted_matrices.shape
    # Create inverse indices for rows and columns
    row_sorted_indices_inv = torch.argsort(row_sorted_indices, dim=1)  # shape: (batch_size, n)
    col_sorted_indices_inv = torch.argsort(col_sorted_indices, dim=1)  # shape: (batch_size, n)
    # Create a batch index tensor
    batch_indices = torch.arange(batch_size)#.unsqueeze(1)  # shape: (batch_size, 1)
    # # Unsort rows and columns using advanced indexing
    unsorted_matrices = sorted_matrices[batch_indices[:,None,None], col_sorted_indices_inv[:,:,None], row_sorted_indices_inv[:,None,:]]
    return unsorted_matrices

# # Example usage
# batch_size, n = 1, 2
# # matrices = torch.rand(batch_size, n, n)
# matrices = F
# sorter = NormSorter()
# sorted_matrices, row_sorted_indices, col_sorted_indices = sort_matrices(matrices)

# print("Original matrices:")
# print(matrices)
# print("\nSorted matrices:")
# print(sorted_matrices)

# # Unsort the matrices
# unsorted_matrices = unsort_matrices(sorted_matrices, row_sorted_indices, col_sorted_indices)
# print("\nUnsorted matrices (should match the original):")
# print(unsorted_matrices)



class GaussianRBF(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.params = params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hparams = params['hparams']
        # self.reset_nontrainableparameters()
        self.reset_parameters()
        
    def reset_parameters(self):
        self.mus = nn.Parameter(torch.Tensor(self.output_dim, self.input_dim))
        self.log_sigmas = nn.Parameter(torch.Tensor(self.output_dim))
        nn.init.uniform_(self.mus, 0, 1)
        nn.init.constant_(self.log_sigmas, np.log((1/self.hparams['h'])**(1/self.params['simparams']['d'])))
                                     
    def reset_nontrainableparameters(self):
        mu_0, mu_1 = np.mgrid[0:1:int(self.hparams['h']**(1/self.params['simparams']['d']))*1j, 0:1:int(self.hparams['h']**(1/self.params['simparams']['d']))*1j]        
        mus = np.vstack([mu_0.ravel(), mu_1.ravel()]).T
        self.mus = torch.tensor(mus)
        self.log_sigmas = torch.ones(self.output_dim)*np.log((1/self.hparams['h']**(1/self.params['simparams']['d'])))               
                                     
    def forward(self, x):
        if self.hparams.get('symgroupavg',False)==True:
            mus_temp = torch.einsum('ij,gj->gi', self.mapping, self.mus - 1/2) + 1/2
            d_scaled = (x[:,None,:] - mus_temp[None,:,:])/torch.exp(self.log_sigmas[None,:,None])
        else:
            d_scaled = (x[:,None,:] - self.mus[None,:,:])/torch.exp(self.log_sigmas[None,:,None])
        y = torch.exp(-torch.sum(d_scaled**2, axis=-1)/2)
        if self.hparams.get('norm_basis',False)==True:
            y = y/torch.sum(y, axis=-1)[:,None]
        return y
    
    def grad(self, x):
        if self.hparams.get('symgroupavg',False)==True:
            mus_temp = torch.einsum('ij,gj->gi', self.mapping, self.mus - 1/2) + 1/2
            prefactor = -1/(torch.exp(self.log_sigmas[None,:,None]))**2*(x[:,None,:] - mus_temp[None,:,:])
        else:
            prefactor = -1/(torch.exp(self.log_sigmas[None,:,None]))**2*(x[:,None,:] - self.mus[None,:,:])
        return prefactor*self.forward(x)[:,:,None]

        
class GaussianRBF_NOMAD(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.hparams = params['hparams']
        
    def forward(self, mus, log_sigmas, x):
        if self.hparams.get('symgroupavg',False)==True:
            mus_temp = torch.einsum('ij,gj->gi', self.mapping, mus - 1/2) + 1/2
            d_scaled = (x[:,:,None,:] - mus_temp[None,None,:,:])/torch.exp(log_sigmas[None,None,:,None])
        else:
            d_scaled = (x[:,:,None,:] - mus[:,None,:,:])/torch.exp(log_sigmas[:,None,:,None])
        y = torch.exp(-torch.sum(d_scaled**2, axis=-1))
        if self.hparams.get('norm_basis',False)==True:
            y = y/torch.sum(y, axis=-1)[:,:,None]
        return y
    

class ReshapeLayer(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape
    
    def forward(self, x):
        new_shape = (x.shape[0],) + self.output_shape
        y = x.reshape(new_shape)
        return y
    
class InversionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.output_shape = output_shape
    
    def forward(self, x):
        y = torch.linalg.pinv(x)
        return y
    
    
class PConv(nn.Module):
    def __init__(self, hidden_channels, kernel_size, stride, bias):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, bias=bias))
        self.layers.append(nn.BatchNorm2d(num_features=hidden_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=1, kernel_size=kernel_size, stride=stride, bias=bias))
    
    def forward(self, x):
        P = x
        for layer in self.layers:
            P = layer(P)
        P = P/torch.norm(P)
        y = x + torch.matmul(P,x)
        return y

    
class UpsampleModel(nn.Module):
    def __init__(self, size):
        super(UpsampleModel, self).__init__()
        self.size = size
        
    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode='nearest')
        return x
    

def expand_D8(A):
    return [A, 
            #A.rot90(dims=[-2,-1]), 
            A.rot90(dims=[-2,-1]).rot90(dims=[-2,-1]), 
            #A.rot90(dims=[-2,-1]).rot90(dims=[-2,-1]).rot90(dims=[-2,-1]),
            A.flip(dims=[-1]),
            #A.flip(dims=[-1]).rot90(dims=[-2,-1]),
            A.flip(dims=[-1]).rot90(dims=[-2,-1]).rot90(dims=[-2,-1])]
            #A.flip(dims=[-1]).rot90(dims=[-2,-1]).rot90(dims=[-2,-1]).rot90(dims=[-2,-1])]