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
        # if self.hparams['outputactivation'] is not None:
        #     self.layers.append(self.hparams['outputactivation'])
        if self.hparams.get('NLB_outputactivation',None) is not None:
            self.layers.append(self.hparams['NLB_outputactivation'])
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
        # if self.hparams['outputactivation'] is not None:
        #     y = self.layers[20](y)
        if self.hparams.get('NLB_outputactivation',None) is not None:
            y = self.layers[20](y)
        if self.hparams.get('outputactivation',None) is not None:
            y = self.layers[20](y)
        return y
    

class CNNNd(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self = customlayers.balance_num_trainable_params(model=self, N_w=self.hparams['N_w'])
        self = self.to(torch.device(self.hparams['used_device']))

    def init_layers(self):
        self.layers = nn.ModuleList()
        self.kernel_sizes = self.hparams['kernel_sizes']
        self.bottleneck_size = self.hparams['bottleneck_size']
        #Layers
        self.layers.append(customlayers.ReshapeLayer(output_shape=self.hparams['input_shape']))
        self.layers.append(customlayers.ConvNd(in_channels=self.hparams['input_shape'][0], out_channels=self.free_parameter, kernel_size=self.kernel_sizes[0], stride=self.kernel_sizes[0], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(customlayers.ConvNd(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[1], stride=self.kernel_sizes[1], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(customlayers.ConvNd(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[2], stride=self.kernel_sizes[2], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(customlayers.ConvNd(in_channels=self.free_parameter, out_channels=self.bottleneck_size, kernel_size=self.kernel_sizes[3], stride=self.kernel_sizes[3], bias=False))
        self.layers.append(customlayers.ReshapeLayer(output_shape=(int(self.bottleneck_size),)))
        self.layers.append(nn.Linear(in_features=int(self.bottleneck_size), out_features=int(self.bottleneck_size)))
        self.layers.append(nn.ReLU())
        self.layers.append(customlayers.ReshapeLayer(output_shape=(int(self.bottleneck_size),1,1)))
        self.layers.append(nn.ConvTranspose2d(self.bottleneck_size, self.free_parameter, kernel_size=self.kernel_sizes[4], stride=self.kernel_sizes[4], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, self.free_parameter, kernel_size=self.kernel_sizes[5], stride=self.kernel_sizes[5], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, self.free_parameter, kernel_size=self.kernel_sizes[6], stride=self.kernel_sizes[6], bias=False))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, 1, kernel_size=self.kernel_sizes[7], stride=self.kernel_sizes[7], bias=False))
        self.layers.append(customlayers.ReshapeLayer(output_shape=self.hparams['output_shape']))           
        # if self.hparams['outputactivation'] is not None:
        #     self.layers.append(self.hparams['outputactivation'])
        if self.hparams.get('NLB_outputactivation',None) is not None:
            self.layers.append(self.hparams['NLB_outputactivation'])
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
        # if self.hparams['outputactivation'] is not None:
        #     y = self.layers[20](y)
        if self.hparams.get('NLB_outputactivation',None) is not None:
            y = self.layers[20](y)
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
        # if self.hparams['outputactivation'] is not None:
        #     self.layers.append(self.hparams['outputactivation'])
        if self.hparams.get('NLB_outputactivation',None) is not None:
            self.layers.append(self.hparams['NLB_outputactivation'])
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
        # if self.hparams['outputactivation'] is not None:
        #     y = self.layers[20](y)
        if self.hparams.get('NLB_outputactivation',None) is not None:
            y = self.layers[20](y)
            print('test1')
        if self.hparams.get('outputactivation',None) is not None:
            y = self.layers[20](y)
            print('test2')
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
        # if self.hparams['outputactivation'] is not None:
        #     self.layers.append(self.hparams['outputactivation'])
        if self.hparams.get('NLB_outputactivation',None) is not None:
            self.layers.append(self.hparams['NLB_outputactivation'])
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
        # if self.hparams['outputactivation'] is not None:
        #     y = self.layers[20](y)
        if self.hparams.get('NLB_outputactivation',None) is not None:
            y = self.layers[20](y)
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
        # bs = 1
        # if x.shape[0]>bs:
        #     y = np.zeros((x.shape[0],)+self.hparams['output_shape'])
        #     for i in range(int(x.shape[0]/bs)):
        #         x_temp = x[bs*i:bs*(i+1)]
        #         for layer in self.layers:
        #             x_temp = layer(x_temp)
        #         y[bs*i:bs*(i+1)] = x_temp.detach().cpu().numpy()
        # if x.shape[0]<=int(self.hparams['batch_size']):
        for layer in self.layers:
            x = layer(x)
        y = x #.detach().cpu().numpy()
        return y    
    

class InvNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self = customlayers.balance_num_trainable_params(model=self, N_w=self.hparams['N_w'])
        
    def init_layers(self):
        self.layers = nn.ModuleList()
        self.kernel_sizes = self.hparams['kernel_sizes']
        self.bottleneck_size = self.hparams['bottleneck_size']
        #Layers
        self.layers.append(customlayers.ReshapeLayer(output_shape=(1,self.hparams['N_F'],self.hparams['N_F'])))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[0], stride=self.kernel_sizes[0], bias=False))
        self.layers.append(nn.Conv2d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[1], stride=self.kernel_sizes[1], bias=False))
        self.layers.append(nn.Conv2d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[2], stride=self.kernel_sizes[2], bias=False))
        self.layers.append(nn.Conv2d(in_channels=self.free_parameter, out_channels=self.bottleneck_size, kernel_size=self.kernel_sizes[3], stride=self.kernel_sizes[3], bias=False))
        self.layers.append(customlayers.InversionLayer())
        self.layers.append(nn.ConvTranspose2d(self.bottleneck_size, self.free_parameter, kernel_size=self.kernel_sizes[4], stride=self.kernel_sizes[4], bias=False))
        #self.layers.append(nn.Tanh())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, self.free_parameter, kernel_size=self.kernel_sizes[5], stride=self.kernel_sizes[5], bias=False))
        #self.layers.append(nn.Tanh())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, self.free_parameter, kernel_size=self.kernel_sizes[6], stride=self.kernel_sizes[6], bias=False))
        #self.layers.append(nn.Tanh())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, 1, kernel_size=self.kernel_sizes[7], stride=self.kernel_sizes[7], bias=False))
        self.layers.append(customlayers.ReshapeLayer(output_shape=(self.hparams['N'],self.hparams['N'])))
        if self.hparams['outputactivation'] is not None:
            self.layers.append(self.hparams['outputactivation'])

    def forward(self, x):
        if self.hparams.get('permutation_equivariance',False)==True:
            x, row_sorted_indices, col_sorted_indices = customlayers.sort_matrices(x)
        if self.hparams.get('scaling_equivariance',False)==True:
            x_norm = torch.amax(torch.abs(x), dim=(-1,-2))
            x = x/x_norm[:,None,None]
        for layer in self.layers:
            x = layer(x)
        y = x
        if self.hparams.get('scaling_equivariance',False)==True:
            y = y/x_norm[:,None,None]    
        if self.hparams.get('permutation_equivariance',False)==True:
            y = customlayers.unsort_matrices(y, row_sorted_indices, col_sorted_indices)
        return y
    

class PNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        #Balancing the number of trainable parameters to N_w
        self.free_parameter = 1
        count = 0
        num_channels_list = []
        while count < self.hparams['N_w']:
            self.init_layers()
            count = sum(p.numel() for p in self.parameters())
            num_channels_list.append(self.free_parameter)
            self.free_parameter +=1
        self.free_parameter = num_channels_list[-2]
        self.init_layers()
        
    def init_layers(self):
        self.layers = nn.ModuleList()
        self.kernel_sizes = self.hparams['kernel_sizes']
        self.bottleneck_size = self.hparams['bottleneck_size']
        #Layers
        self.layers.append(customlayers.ReshapeLayer(output_shape=(1,self.hparams['h_F'][0],self.hparams['h_F'][1])) if self.hparams['modeltype']=='data NGO' else customlayers.ReshapeLayer(output_shape=(1,self.hparams['N_F'],self.hparams['N_F'])))
        self.layers.append(nn.ConvTranspose2d(1, self.free_parameter, kernel_size=self.kernel_sizes[0], stride=self.kernel_sizes[0], bias=self.hparams.get('bias_NLBranch', True)))
        # self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, self.free_parameter, kernel_size=self.kernel_sizes[1], stride=self.kernel_sizes[1], bias=False))
        # self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, self.free_parameter, kernel_size=self.kernel_sizes[2], stride=self.kernel_sizes[2], bias=False))
        # self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.ConvTranspose2d(self.free_parameter, 1, kernel_size=self.kernel_sizes[3], stride=self.kernel_sizes[3], bias=False))
        self.layers.append(customlayers.ReshapeLayer(output_shape=(self.hparams['N'],self.hparams['N'])))
        if self.hparams['outputactivation'] is not None:
            self.layers.append(self.hparams['outputactivation'])

    def forward(self, x):
        if self.hparams.get('permutation_equivariance',False)==True:
            x, row_sorted_indices, col_sorted_indices = customlayers.sort_matrices(x)
        if self.hparams.get('scaling_equivariance',False)==True:
            x_norm = torch.amax(torch.abs(x), dim=(-1,-2))
            x = x/x_norm[:,None,None]
        for layer in self.layers:
            x = layer(x)
        y = x
        if self.hparams.get('scaling_equivariance',False)==True:
            y = y/x_norm[:,None,None]    
        if self.hparams.get('permutation_equivariance',False)==True:
            y = customlayers.unsort_matrices(y, row_sorted_indices, col_sorted_indices)
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
    
    
class SPCNN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.layers = nn.ModuleList()
        self.kernel_size = self.hparams['kernel_sizes'][0]
        self.free_parameter =self.hparams['free_parameter']
    
        # Adjusted convolutional layers
        self.layers.append(customlayers.ReshapeLayer(output_shape=(1,self.hparams['N'],self.hparams['N'])))
        self.layers.append(nn.ConvTranspose2d(in_channels=1, out_channels=self.free_parameter, kernel_size=self.kernel_size, stride=1, bias=False))
        self.layers.append(nn.LeakyReLU())
        
        # #self.layers.append(nn.BatchNorm2d(num_features=self.free_parameter, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_size, stride=1, bias=False))
        self.layers.append(nn.LeakyReLU())
        # #self.layers.append(nn.BatchNorm2d(num_features=self.free_parameter, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_size, stride=1, bias=False))
        self.layers.append(nn.LeakyReLU())
        # #self.layers.append(nn.BatchNorm2d(num_features=self.free_parameter, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        # self.layers.append(nn.Conv2d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_size, stride=1, bias=False))
        # self.layers.append(nn.LeakyReLU())
        # #self.layers.append(nn.BatchNorm2d(num_features=self.free_parameter, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        # self.layers.append(nn.ConvTranspose2d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_size, stride=1, bias=False))
        # self.layers.append(nn.LeakyReLU())
        # #self.layers.append(nn.BatchNorm2d(num_features=self.free_parameter, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        # self.layers.append(nn.Conv2d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_size, stride=1, bias=False))
        # self.layers.append(nn.LeakyReLU())
        # #self.layers.append(nn.BatchNorm2d(num_features=self.free_parameter, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        # self.layers.append(nn.ConvTranspose2d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_size, stride=1, bias=False))
        # self.layers.append(nn.LeakyReLU())

        #self.layers.append(nn.BatchNorm2d(num_features=self.free_parameter, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=self.free_parameter, out_channels=1, kernel_size=self.kernel_size, stride=1, bias=False))
        self.layers.append(customlayers.ReshapeLayer(output_shape=(64,64)))
        if self.hparams['outputactivation'] is not None:
            self.layers.append(self.hparams['outputactivation'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        y = x
        return y
    

class AcCNN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        #Balancing the number of trainable parameters to N_w
        self.free_parameter = 1
        count = 0
        num_channels_list = []
        while count < self.hparams['N_w']:
            self.init_layers()
            count = sum(p.numel() for p in self.parameters())
            num_channels_list.append(self.free_parameter)
            self.free_parameter +=1
        self.free_parameter = num_channels_list[-2]
        self.init_layers()
        
    def init_layers(self):
        self.layers = nn.ModuleList()
        self.kernel_sizes = self.hparams['kernel_sizes']
        self.bottleneck_size = self.hparams['bottleneck_size']
        #Layers
        self.layers.append(customlayers.ReshapeLayer(output_shape=(1,self.hparams['h'][0],self.hparams['h'][1])) if self.hparams['model/data']=='data' else customlayers.ReshapeLayer(output_shape=(1,self.hparams['N'],self.hparams['N'])))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[0], stride=self.kernel_sizes[0], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        # self.layers.append(nn.BatchNorm2d(num_features=self.free_parameter, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[1], stride=self.kernel_sizes[1], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        # self.layers.append(nn.BatchNorm2d(num_features=self.free_parameter, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[2], stride=self.kernel_sizes[2], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        # self.layers.append(nn.BatchNorm2d(num_features=self.free_parameter, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=self.free_parameter, out_channels=self.free_parameter, kernel_size=self.kernel_sizes[3], stride=self.kernel_sizes[3], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(customlayers.ReshapeLayer(output_shape=(int(self.free_parameter),)))
        self.layers.append(nn.Linear(in_features=int(self.free_parameter), out_features=int(self.free_parameter)))
        self.layers.append(nn.LeakyReLU())
        # self.layers.append(nn.Linear(in_features=int(self.free_parameter), out_features=int(self.free_parameter)))
        # self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(in_features=int(self.free_parameter), out_features=int(self.hparams['k'])))
        if self.hparams['outputactivation'] is not None:
            self.layers.append(self.hparams['outputactivation'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        y = x
        return y
    

class KandK(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.init_layers()
        
    def init_layers(self):
        self.layers1 = nn.ModuleList()
        #Layers
        for i in range(self.hparams['N_layers']):
            self.layers1.append(nn.Linear(in_features=int(self.hparams['N']), out_features=int(self.hparams['k']), bias=True))
            self.layers1.append(nn.Softmax()) 
            self.layers1.append(nn.Linear(in_features=int(self.hparams['k']), out_features=int(self.hparams['k']), bias=False)) 

        # self.layers2 = nn.ModuleList()
        # for i in range(self.hparams['N_layers']):
        #     self.layers2.append(nn.Linear(in_features=int(self.hparams['k']), out_features=int(100), bias=False)) 
        #     self.layers2.append(nn.ReLU())
        #     self.layers2.append(nn.Linear(in_features=int(100), out_features=int(100), bias=False)) 
        #     self.layers2.append(nn.ReLU())
        #     self.layers2.append(nn.Linear(in_features=int(100), out_features=int(100), bias=False)) 
        #     self.layers2.append(nn.ReLU())
        #     self.layers2.append(nn.Linear(in_features=int(100), out_features=int(100), bias=False)) 
        #     self.layers2.append(nn.ReLU())
        #     self.layers2.append(nn.Linear(in_features=int(100), out_features=int(self.hparams['k']), bias=False)) 

    def forward(self, x):
        F = x
        F_i = x
        lnabsTrF_i = torch.zeros((x.shape[0],self.hparams['N']), dtype=self.hparams['dtype'], device=self.device)
        TrF_i = opt_einsum.contract('nii->n', F_i)
        lnabsTrF_i[:,0] = torch.log(torch.abs(TrF_i))
        for i in range(1,self.hparams['N']):
            F_i = torch.matmul(F,F_i)
            # F_i = F_i/torch.amax(torch.abs(F_i), dim=(-1,-2))[:,None,None]
            TrF_i = opt_einsum.contract('nii->n', F_i)
            lnabsTrF_i[:,i] = torch.log(torch.abs(TrF_i))
        c = torch.zeros(x.shape[0],self.hparams['k'], dtype=self.hparams['dtype'], device=self.device)
        for i in range(0, len(self.layers1), 3):
            c_i = torch.exp(self.layers1[i](lnabsTrF_i))
            c_i = self.layers1[i+1](c_i)
            c_i = self.layers1[i+2](c_i)
            c += c_i
        # for layer in self.layers2:
        #     c = layer(c)
        return c
    

class DeepONetBranch(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim=2*self.hparams['Q']**self.hparams['d']+4*self.hparams['Q'], output_dim=self.hparams['N'], bias=False))
        if self.hparams['outputactivation']!=None:
            self.layers.append(self.hparams['outputactivation'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            y = x
        return y