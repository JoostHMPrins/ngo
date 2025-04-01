# Copyright 2025 Joost Prins

# 3rd Party
import numpy as np
import torch
import torch.nn as nn
import opt_einsum
import neuralop

# Local
import ngo.ml.customlayers as customlayers


class MLP(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self = customlayers.balance_num_trainable_params(model=self, N_w=self.hparams['N_w'])
        
    def init_layers(self):
        self.layers = nn.ModuleList()
        self.kernel_sizes = self.hparams['kernel_sizes']
        #Layers
        self.layers.append(customlayers.ReshapeLayer(output_shape=(int(np.prod(self.hparams['input_shape'])),)))
        self.layers.append(nn.Linear(in_features=int(np.prod(self.hparams['input_shape'])), out_features=self.free_parameter, bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(in_features=self.free_parameter, out_features=self.free_parameter, bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(in_features=self.free_parameter, out_features=self.free_parameter, bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(in_features=self.free_parameter, out_features=int(np.prod(self.hparams['output_shape'])), bias=False))
        self.layers.append(customlayers.ReshapeLayer(output_shape=self.hparams['output_shape']))        
        if self.hparams['outputactivation'] is not None:
            self.layers.append(self.hparams['outputactivation'])

    def forward(self, x):
        x0 = x
        x1 = self.layers[0](x0)
        x2 = self.layers[1](x1)
        x3 = self.layers[2](x2)
        x4 = self.layers[3](x3)
        x5 = self.layers[4](x4) + x3 if self.hparams['skipconnections']==True else self.layers[4](x4)
        x6 = self.layers[5](x5)
        x7 = self.layers[6](x6) + x5 if self.hparams['skipconnections']==True else self.layers[6](x6)
        x8 = self.layers[7](x7)
        y = self.layers[8](x8)
        if self.hparams['outputactivation'] is not None:
            y = self.layers[9](y)
        return y


class CNN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self = customlayers.balance_num_trainable_params(model=self, N_w=self.hparams['N_w'])
        
    def init_layers(self):
        self.layers = nn.ModuleList()
        self.kernel_sizes = self.hparams['kernel_sizes']
        self.bottleneck_size = self.hparams['bottleneck_size']
        #Layers
        self.layers.append(customlayers.ReshapeLayer(output_shape=self.hparams['input_shape']))
        self.layers.append(nn.Conv2d(in_channels=self.hparams['input_shape'][0], out_channels=self.free_parameter, kernel_size=self.kernel_sizes[0], stride=self.kernel_sizes[0], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[1], stride=self.kernel_sizes[1], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[2], stride=self.kernel_sizes[2], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=self.free_parameter, out_channels=self.bottleneck_size, kernel_size=self.kernel_sizes[3], stride=self.kernel_sizes[3], bias=False))
        self.layers.append(customlayers.ReshapeLayer(output_shape=(int(self.bottleneck_size),)))
        self.layers.append(nn.Linear(in_features=int(self.bottleneck_size), out_features=int(self.bottleneck_size)))
        self.layers.append(nn.ReLU())
        self.layers.append(customlayers.ReshapeLayer(output_shape=(self.bottleneck_size,1,1)))
        self.layers.append(nn.ConvTranspose2d(self.bottleneck_size, self.free_parameter, kernel_size=self.kernel_sizes[4], stride=self.kernel_sizes[4], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, self.free_parameter, kernel_size=self.kernel_sizes[5], stride=self.kernel_sizes[5], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, self.free_parameter, kernel_size=self.kernel_sizes[6], stride=self.kernel_sizes[6], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, 1, kernel_size=self.kernel_sizes[7], stride=self.kernel_sizes[7], bias=False))
        self.layers.append(customlayers.ReshapeLayer(output_shape=self.hparams['output_shape']))           
        if self.hparams.get('outputactivation',None) is not None:
            self.layers.append(self.hparams['outputactivation'])

    def forward(self, x):
        x0 = x
        x1 = self.layers[0](x0)
        x2 = self.layers[1](x1)
        x3 = self.layers[2](x2)
        x4 = self.layers[3](x3)
        x5 = self.layers[4](x4)
        x6 = self.layers[5](x5)
        x7 = self.layers[6](x6)
        x8 = self.layers[7](x7)
        x9 = self.layers[8](x8)
        x10 = self.layers[9](x9)
        x11 = self.layers[10](x10) + x9 if self.hparams['skipconnections']==True else self.layers[10](x10)
        x12 = self.layers[11](x11)
        x13 = self.layers[12](x12)
        x14 = self.layers[13](x13) + x7 if self.hparams['skipconnections']==True else self.layers[13](x13)
        x15 = self.layers[14](x14)
        x16 = self.layers[15](x15) + x5 if self.hparams['skipconnections']==True else self.layers[15](x15)
        x17 = self.layers[16](x16)
        x18 = self.layers[17](x17) + x3 if self.hparams['skipconnections']==True else self.layers[17](x17)
        x19 = self.layers[18](x18) + x1 if self.hparams['modeltype']=='model NGO' else self.layers[18](x18)
        x20 = self.layers[19](x19)
        y = x20
        if self.hparams.get('outputactivation',None) is not None:
            y = self.layers[20](y)
        return y
    

class CNN_3dto2d(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self = customlayers.balance_num_trainable_params(model=self, N_w=self.hparams['N_w'])
        
    def init_layers(self):
        self.layers = nn.ModuleList()
        self.kernel_sizes = self.hparams['kernel_sizes']
        self.bottleneck_size = self.hparams['bottleneck_size']
        #Layers
        self.layers.append(customlayers.ReshapeLayer(output_shape=self.hparams['input_shape']))
        self.layers.append(nn.Conv3d(in_channels=self.hparams['input_shape'][0], out_channels=self.free_parameter, kernel_size=self.kernel_sizes[0], stride=self.kernel_sizes[0], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv3d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[1], stride=self.kernel_sizes[1], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv3d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[2], stride=self.kernel_sizes[2], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv3d(in_channels=self.free_parameter, out_channels=self.bottleneck_size, kernel_size=self.kernel_sizes[3], stride=self.kernel_sizes[3], bias=False))
        self.layers.append(customlayers.ReshapeLayer(output_shape=(int(self.bottleneck_size),)))
        self.layers.append(nn.Linear(in_features=int(self.bottleneck_size), out_features=int(self.bottleneck_size)))
        self.layers.append(nn.ReLU())
        self.layers.append(customlayers.ReshapeLayer(output_shape=(self.bottleneck_size,1,1)))
        self.layers.append(nn.ConvTranspose2d(self.bottleneck_size, self.free_parameter, kernel_size=self.kernel_sizes[4], stride=self.kernel_sizes[4], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, self.free_parameter, kernel_size=self.kernel_sizes[5], stride=self.kernel_sizes[5], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, self.free_parameter, kernel_size=self.kernel_sizes[6], stride=self.kernel_sizes[6], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, 1, kernel_size=self.kernel_sizes[7], stride=self.kernel_sizes[7], bias=False))
        self.layers.append(customlayers.ReshapeLayer(output_shape=self.hparams['output_shape']))           
        if self.hparams.get('outputactivation',None) is not None:
            self.layers.append(self.hparams['outputactivation'])

    def forward(self, x):
        x0 = x
        x1 = self.layers[0](x0)
        x2 = self.layers[1](x1)
        x3 = self.layers[2](x2)
        x4 = self.layers[3](x3)
        x5 = self.layers[4](x4)
        x6 = self.layers[5](x5)
        x7 = self.layers[6](x6)
        x8 = self.layers[7](x7)
        x9 = self.layers[8](x8)
        x10 = self.layers[9](x9)
        x11 = self.layers[10](x10) + x9 if self.hparams['skipconnections']==True else self.layers[10](x10)
        x12 = self.layers[11](x11)
        x13 = self.layers[12](x12)
        x14 = self.layers[13](x13) + x7 if self.hparams['skipconnections']==True else self.layers[13](x13)
        x15 = self.layers[14](x14)
        x16 = self.layers[15](x15) + x5 if self.hparams['skipconnections']==True else self.layers[15](x15)
        x17 = self.layers[16](x16)
        x18 = self.layers[17](x17) + x3 if self.hparams['skipconnections']==True else self.layers[17](x17)
        x19 = self.layers[18](x18) + x1 if self.hparams['modeltype']=='model NGO' else self.layers[18](x18)
        x20 = self.layers[19](x19)
        y = x20
        if self.hparams.get('outputactivation',None) is not None:
            y = self.layers[20](y)
        return y
    

class CNN_3dto3d(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self = customlayers.balance_num_trainable_params(model=self, N_w=self.hparams['N_w'])
        
    def init_layers(self):
        self.layers = nn.ModuleList()
        self.kernel_sizes = self.hparams['kernel_sizes']
        self.bottleneck_size = self.hparams['bottleneck_size']
        #Layers
        self.layers.append(customlayers.ReshapeLayer(output_shape=self.hparams['input_shape']))
        self.layers.append(nn.Conv3d(in_channels=self.hparams['input_shape'][0], out_channels=self.free_parameter, kernel_size=self.kernel_sizes[0], stride=self.kernel_sizes[0], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv3d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[1], stride=self.kernel_sizes[1], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv3d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[2], stride=self.kernel_sizes[2], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv3d(in_channels=self.free_parameter, out_channels=self.bottleneck_size, kernel_size=self.kernel_sizes[3], stride=self.kernel_sizes[3], bias=False))
        self.layers.append(customlayers.ReshapeLayer(output_shape=(int(self.bottleneck_size),)))
        self.layers.append(nn.Linear(in_features=int(self.bottleneck_size), out_features=int(self.bottleneck_size)))
        self.layers.append(nn.ReLU())
        self.layers.append(customlayers.ReshapeLayer(output_shape=(self.bottleneck_size,1,1,1)))
        self.layers.append(nn.ConvTranspose3d(self.bottleneck_size, self.free_parameter, kernel_size=self.kernel_sizes[4], stride=self.kernel_sizes[4], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose3d(self.free_parameter, self.free_parameter, kernel_size=self.kernel_sizes[5], stride=self.kernel_sizes[5], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose3d(self.free_parameter, self.free_parameter, kernel_size=self.kernel_sizes[6], stride=self.kernel_sizes[6], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose3d(self.free_parameter, 1, kernel_size=self.kernel_sizes[7], stride=self.kernel_sizes[7], bias=False))
        self.layers.append(customlayers.ReshapeLayer(output_shape=self.hparams['output_shape']))           
        if self.hparams.get('outputactivation',None) is not None:
            self.layers.append(self.hparams['outputactivation'])

    def forward(self, x):
        x0 = x
        x1 = self.layers[0](x0)
        x2 = self.layers[1](x1)
        x3 = self.layers[2](x2)
        x4 = self.layers[3](x3)
        x5 = self.layers[4](x4)
        x6 = self.layers[5](x5)
        x7 = self.layers[6](x6)
        x8 = self.layers[7](x7)
        x9 = self.layers[8](x8)
        x10 = self.layers[9](x9)
        x11 = self.layers[10](x10) + x9 if self.hparams['skipconnections']==True else self.layers[10](x10)
        x12 = self.layers[11](x11)
        x13 = self.layers[12](x12)
        x14 = self.layers[13](x13) + x7 if self.hparams['skipconnections']==True else self.layers[13](x13)
        x15 = self.layers[14](x14)
        x16 = self.layers[15](x15) + x5 if self.hparams['skipconnections']==True else self.layers[15](x15)
        x17 = self.layers[16](x16)
        x18 = self.layers[17](x17) + x3 if self.hparams['skipconnections']==True else self.layers[17](x17)
        x19 = self.layers[18](x18) + x1 if self.hparams['modeltype']=='model NGO' else self.layers[18](x18)
        x20 = self.layers[19](x19)
        y = x20
        if self.hparams.get('outputactivation',None) is not None:
            y = self.layers[20](y)
        return y
    

class FNO(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self = customlayers.balance_num_trainable_params(model=self, N_w=self.hparams['N_w'])

    def init_layers(self):
        self.layers = nn.ModuleList()
        self.layers.append(customlayers.ReshapeLayer(output_shape=self.hparams['input_shape']))
        self.layers.append(neuralop.models.FNO(n_modes=self.hparams['h'], in_channels=self.hparams['input_shape'][0], out_channels=1, hidden_channels=self.free_parameter, domain_padding=0.2))
        self.layers.append(customlayers.ReshapeLayer(output_shape=self.hparams['output_shape']))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        y = x
        return y    


class LBranchNet(nn.Module):
    def __init__(self, hparams, input_dim, output_dim):
        super().__init__()
        self.hparams = hparams
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, output_dim, bias=False))

    def forward(self, x):
        if self.hparams.get('scaling_equivariance',False)==True:
            x_norm = torch.amax(torch.abs(x), dim=(-1,-2))
            x = x/x_norm[:,None,None]
        for layer in self.layers:
            x = layer(x)
            y = x
        if self.hparams.get('scaling_equivariance',False)==True:
            y = y*x_norm[:,None]
        return y