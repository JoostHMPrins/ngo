import torch
from torch import nn
from customlayers import *
    
    
class UNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        #Balancing the number of trainable parameters to N_w
        self.num_channels = 1
        count = 0
        num_channels_list = []
        while count < self.hparams['N_w']:
            self.init_layers()
            count = sum(p.numel() for p in self.parameters())
            num_channels_list.append(self.num_channels)
            self.num_channels +=1
        self.num_channels = num_channels_list[-2]
        self.init_layers()
        
    def init_layers(self):
        self.layers = nn.ModuleList()
        self.kernel_sizes = self.hparams['kernel_sizes']
        self.bottleneck_size = self.hparams['bottleneck_size']
        #Layers

        self.layers.append(ReshapeLayer(output_shape=(1,self.hparams['h'][0],self.hparams['h'][1])) if self.hparams['model/data']=='data' else ReshapeLayer(output_shape=(1,self.hparams['N'],self.hparams['N'])))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=self.kernel_sizes[0], stride=self.kernel_sizes[0], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_sizes[1], stride=self.kernel_sizes[1], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_sizes[2], stride=self.kernel_sizes[2], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=self.bottleneck_size, kernel_size=self.kernel_sizes[3], stride=self.kernel_sizes[3], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer(output_shape=(int(self.bottleneck_size),)))
        self.layers.append(nn.Linear(in_features=int(self.bottleneck_size), out_features=int(self.bottleneck_size)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(ReshapeLayer(output_shape=(self.bottleneck_size,1,1)))
        self.layers.append(nn.ConvTranspose2d(self.bottleneck_size, self.num_channels, kernel_size=self.kernel_sizes[4], stride=self.kernel_sizes[4], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.ConvTranspose2d(self.num_channels, self.num_channels, kernel_size=self.kernel_sizes[5], stride=self.kernel_sizes[5], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.ConvTranspose2d(self.num_channels, self.num_channels, kernel_size=self.kernel_sizes[6], stride=self.kernel_sizes[6], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.ConvTranspose2d(self.num_channels, 1, kernel_size=self.kernel_sizes[7], stride=self.kernel_sizes[7], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer(output_shape=(self.hparams['N'],self.hparams['N'])))
        if self.hparams['NLB_outputactivation'] is not None:
            self.layers.append(self.hparams['NLB_outputactivation'])

    def forward(self, x):
        if self.hparams.get('scaling_equivariance',False)==True:
            x_norm = torch.amax(torch.abs(x), dim=(-1,-2))
            x = x/x_norm[:,None,None]
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
        x11 = self.layers[10](x10) + x9
        x12 = self.layers[11](x11)
        x13 = self.layers[12](x12)
        x14 = self.layers[13](x13) + x7
        x15 = self.layers[14](x14)
        x16 = self.layers[15](x15) + x5
        x17 = self.layers[16](x16)
        x18 = self.layers[17](x17) + x3
        x19 = self.layers[18](x18) if self.hparams['model/data']=='data' else self.layers[18](x18) + x1
        x20 = self.layers[19](x19)
        x21 = self.layers[20](x20)
        y = x21
        if self.hparams.get('scaling_equivariance',False)==True:
            y = y/x_norm[:,None,None]    
        return y


class NLBranch_NGO(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        
        # Adjusted convolutional layers
        self.layers.append(ReshapeLayer(output_shape=(1,self.hparams['N'],self.hparams['N'])))
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
    
    
class InvCNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        self.channels = 1
        
        # Adjusted convolutional layers
        self.layers.append(ReshapeLayer(output_shape=(1,self.hparams['N'],self.hparams['N'])))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=self.channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=self.channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=self.channels, out_channels=1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer(output_shape=(self.channels,int(self.hparams['N']/16),int(self.hparams['N']/16))))
        self.layers.append(nn.ReLU())
        self.layers.append(InversionLayer())
        self.layers.append(ReshapeLayer(output_shape=(self.channels, int(self.hparams['N']/16),int(self.hparams['N']/16))))
        self.layers.append(nn.ConvTranspose2d(1, self.channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=self.channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(self.channels, self.channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=self.channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(self.channels, self.channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(self.channels, 1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer(output_shape=(self.hparams['N'],self.hparams['N'])))
        if self.hparams['NLB_outputactivation'] is not None:
            self.layers.append(self.hparams['NLB_outputactivation'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        y = x
        return y
    
    
class SPCNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        self.kernel_size = self.hparams['kernel_sizes'][0]
        self.num_channels =self.hparams['num_channels']
    
        # Adjusted convolutional layers
        self.layers.append(ReshapeLayer(output_shape=(1,self.hparams['N'],self.hparams['N'])))
        self.layers.append(nn.ConvTranspose2d(in_channels=1, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, bias=False))
        self.layers.append(nn.LeakyReLU())
        
        # #self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, bias=False))
        self.layers.append(nn.LeakyReLU())
        # #self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, bias=False))
        self.layers.append(nn.LeakyReLU())
        # #self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        # self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, bias=False))
        # self.layers.append(nn.LeakyReLU())
        # #self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        # self.layers.append(nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, bias=False))
        # self.layers.append(nn.LeakyReLU())
        # #self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        # self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, bias=False))
        # self.layers.append(nn.LeakyReLU())
        # #self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        # self.layers.append(nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, bias=False))
        # self.layers.append(nn.LeakyReLU())

        #self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=1, kernel_size=self.kernel_size, stride=1, bias=False))
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
        if self.hparams.get('scaling_equivariance',False)==True:
            x_norm = torch.amax(torch.abs(x), dim=(-1,-2))
            x = x/x_norm[:,None,None]
        for layer in self.layers:
            x = layer(x)
            y = x
        if self.hparams.get('scaling_equivariance',False)==True:
            y = y*x_norm[:,None]
        return y
    

class DeepONetBranch(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim=2*self.hparams['Q']**params['simparams']['d']+4*self.hparams['Q'], output_dim=self.hparams['N'], bias=self.hparams.get('bias_NLBranch',True)))
        if self.hparams['NLB_outputactivation']!=None:
            self.layers.append(self.hparams['NLB_outputactivation'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            y = x
        return y