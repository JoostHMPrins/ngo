import torch
from torch import nn
from customlayers import *

class NLBranch_NGO(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        
        # Adjusted convolutional layers
        self.layers.append(ReshapeLayer(output_shape=(1,self.hparams['h'],self.hparams['h'])))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=16, out_channels=1, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer(output_shape=(100,)))
        self.layers.append(nn.Linear(100,100))
        self.layers.append(nn.ReLU())
        self.layers.append(ReshapeLayer(output_shape=(1,10,10)))
        self.layers.append(nn.ConvTranspose2d(1, 16, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(16, 32, kernel_size=4, stride=1, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer(output_shape=(64,64)))
        if self.hparams['NLB_outputactivation'] is not None:
            self.layers.append(self.hparams['NLB_outputactivation'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        y = x
        return y


class NLBranch_VarMiON(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        
        # Adjusted convolutional layers
        self.layers.append(ReshapeLayer(output_shape=(1,12,12)))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=4, out_channels=16, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer(output_shape=(32,)))
        self.layers.append(nn.Linear(32,32))
        self.layers.append(ReshapeLayer(output_shape=(32,1,1)))
        self.layers.append(nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(16, 8, kernel_size=4, stride=4, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer(output_shape=(64,64)))
        if self.hparams['NLB_outputactivation'] is not None:
            self.layers.append(self.hparams['NLB_outputactivation'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        y = x
        return y
    
    
class LBranchNet(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, output_dim, bias=self.hparams.get('bias_LBranch',True)))

    def forward(self, x):
        if self.hparams.get('scale_invariance',False)==True:
            x_norm = torch.amax(torch.abs(x), dim=(-1,-2))
            x = x/x_norm[:,None,None]
        for layer in self.layers:
            x = layer(x)
            y = x
        if self.hparams.get('scale_invariance',False)==True:
            y = y*x_norm[:,None]
        return y
    

class DeepONetBranch(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim,output_dim, bias=self.hparams.get('bias_NLBranch',True)))
        if self.hparams['NLB_outputactivation']!=None:
            self.layers.append(self.hparams['NLB_outputactivation'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            y = x
        return y