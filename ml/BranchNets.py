import torch
from torch import nn
from customlayers import *


# class UNet(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self.hparams = params['hparams']
#         self.layers = nn.ModuleList()
#         self.num_channels = self.hparams['num_channels']
#         self.kernel_sizes = self.hparams['kernel_sizes']
#         self.compression = self.kernel_sizes[0]*self.kernel_sizes[1]*self.kernel_sizes[2]*self.kernel_sizes[3]
#         # Adjusted convolutional layers
#         self.layers.append(ReshapeLayer(output_shape=(1,self.hparams['h'],self.hparams['h'])))
#         self.layers.append(nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=self.kernel_sizes[0], stride=self.kernel_sizes[0], bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
#         self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_sizes[1], stride=self.kernel_sizes[1], bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
#         self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_sizes[2], stride=self.kernel_sizes[2], bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_sizes[3], stride=self.kernel_sizes[3], bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(ReshapeLayer(output_shape=(int(self.num_channels*(self.hparams['h']/self.compression)**2),)))
#         self.layers.append(nn.Linear(in_features=int(self.num_channels*(self.hparams['h']/self.compression)**2),out_features=int(self.num_channels*(self.hparams['h']/self.compression)**2)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(ReshapeLayer(output_shape=(self.num_channels,int(self.hparams['h']/self.compression),int(self.hparams['h']/self.compression))))
#         self.layers.append(nn.ConvTranspose2d(self.num_channels, self.num_channels, kernel_size=self.kernel_sizes[3], stride=self.kernel_sizes[3], bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
#         self.layers.append(nn.ConvTranspose2d(self.num_channels, self.num_channels, kernel_size=self.kernel_sizes[2], stride=self.kernel_sizes[2], bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
#         self.layers.append(nn.ConvTranspose2d(self.num_channels, self.num_channels, kernel_size=self.kernel_sizes[1], stride=self.kernel_sizes[1], bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.ConvTranspose2d(self.num_channels, 1, kernel_size=self.kernel_sizes[0], stride=self.kernel_sizes[0], bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(ReshapeLayer(output_shape=(self.hparams['h'],self.hparams['h'])))
#         if self.hparams['NLB_outputactivation'] is not None:
#             self.layers.append(self.hparams['NLB_outputactivation'])

            
#     def forward(self, x):
#         x0 = x
#         x1 = self.layers[0](x0)
#         x2 = self.layers[1](x1)
#         x3 = self.layers[2](x2)
#         x4 = self.layers[3](x3)
#         x5 = self.layers[4](x4)
#         x6 = self.layers[5](x5)
#         x7 = self.layers[6](x6)
#         x8 = self.layers[7](x7)
#         x9 = self.layers[8](x8)
#         x10 = self.layers[9](x9) 
#         x11 = self.layers[10](x10)
#         x12 = self.layers[11](x11)
#         x13 = self.layers[12](x12) + x11
#         x14 = self.layers[13](x13)
#         x15 = self.layers[14](x14)
#         x16 = self.layers[15](x15) + x9
#         x17 = self.layers[16](x16)
#         x18 = self.layers[17](x17)
#         x19 = self.layers[18](x18) + x6
#         x20 = self.layers[19](x19)
#         x21 = self.layers[20](x20) 
#         x22 = self.layers[21](x21) + x3
#         x23 = self.layers[22](x22) + x1
#         x24 = self.layers[23](x23)
#         x25 = self.layers[24](x24)
#         y = x25
#         return y
    
    
class UNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        self.num_channels = self.hparams['num_channels']
        self.kernel_sizes = self.hparams['kernel_sizes']
        self.compression = self.kernel_sizes[0]*self.kernel_sizes[1]*self.kernel_sizes[2]*self.kernel_sizes[3]
        # Adjusted convolutional layers
        self.layers.append(ReshapeLayer(output_shape=(1,self.hparams['h'],self.hparams['h'])))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=self.kernel_sizes[0], stride=self.kernel_sizes[0], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_sizes[1], stride=self.kernel_sizes[1], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_sizes[2], stride=self.kernel_sizes[2], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_sizes[3], stride=self.kernel_sizes[3], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer(output_shape=(int(self.num_channels*(self.hparams['h']/self.compression)**2),)))
        self.layers.append(nn.Linear(in_features=int(self.num_channels*(self.hparams['h']/self.compression)**2),out_features=int(self.num_channels*(self.hparams['h']/self.compression)**2)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(ReshapeLayer(output_shape=(self.num_channels,int(self.hparams['h']/self.compression),int(self.hparams['h']/self.compression))))
        self.layers.append(nn.ConvTranspose2d(self.num_channels, self.num_channels, kernel_size=self.kernel_sizes[3], stride=self.kernel_sizes[3], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.ConvTranspose2d(self.num_channels, self.num_channels, kernel_size=self.kernel_sizes[2], stride=self.kernel_sizes[2], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.ConvTranspose2d(self.num_channels, self.num_channels, kernel_size=self.kernel_sizes[1], stride=self.kernel_sizes[1], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.ConvTranspose2d(self.num_channels, 1, kernel_size=self.kernel_sizes[0], stride=self.kernel_sizes[0], bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer(output_shape=(self.hparams['h'],self.hparams['h'])))
        if self.hparams['NLB_outputactivation'] is not None:
            self.layers.append(self.hparams['NLB_outputactivation'])

            
    def forward(self, x):
        if self.hparams.get('scale_invariance',False)==True:
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
        x11 = self.layers[10](x10) + x9 if self.hparams['UNet']==True else self.layers[10](x10)
        x12 = self.layers[11](x11)
        x13 = self.layers[12](x12)
        x14 = self.layers[13](x13) + x7 if self.hparams['UNet']==True else self.layers[13](x13)
        x15 = self.layers[14](x14)
        x16 = self.layers[15](x15) + x5 if self.hparams['UNet']==True else self.layers[15](x15)
        x17 = self.layers[16](x16)
        x18 = self.layers[17](x17) + x3 if self.hparams['UNet']==True else self.layers[17](x17)
        x19 = self.layers[18](x18) + x1 if self.hparams['UNet']==True else self.layers[18](x18)
        x20 = self.layers[19](x19)
        x21 = self.layers[20](x20)
        y = x21
        if self.hparams.get('scale_invariance',False)==True:
            y = y/x_norm[:,None,None]    
        return y



# class UNet(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self.hparams = params['hparams']
#         self.layers = nn.ModuleList()
#         self.num_channels = self.hparams['num_channels']
#         # Adjusted convolutional layers
#         self.layers.append(ReshapeLayer(output_shape=(1,self.hparams['h'],self.hparams['h'])))
#         self.layers.append(nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=2*self.num_channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.Conv2d(in_channels=2*self.num_channels, out_channels=1*self.num_channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.Conv2d(in_channels=1*self.num_channels, out_channels=1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(ReshapeLayer(output_shape=(int(1*(self.hparams['h']/16)**2),)))
#         self.layers.append(nn.Linear(in_features=int(1*(self.hparams['h']/16)**2),out_features=int(1*(self.hparams['h']/16)**2)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(ReshapeLayer(output_shape=(1,int(self.hparams['h']/16),int(self.hparams['h']/16))))
#         self.layers.append(nn.ConvTranspose2d(1, self.num_channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.ConvTranspose2d(self.num_channels, 2*self.num_channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.ConvTranspose2d(2*self.num_channels, 1*self.num_channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.ConvTranspose2d(1*self.num_channels, 1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(ReshapeLayer(output_shape=(self.hparams['h'],self.hparams['h'])))
#         if self.hparams['NLB_outputactivation'] is not None:
#             self.layers.append(self.hparams['NLB_outputactivation'])

#     def forward(self, x):
#         x0 = x
#         x1 = self.layers[0](x0)
#         x2 = self.layers[1](x1)
#         x3 = self.layers[2](x2)
#         x4 = self.layers[3](x3)
#         x5 = self.layers[4](x4)
#         x6 = self.layers[5](x5)
#         x7 = self.layers[6](x6)
#         x8 = self.layers[7](x7)
#         x9 = self.layers[8](x8)
#         x10 = self.layers[9](x9)
#         x11 = self.layers[10](x10) + x9 if self.hparams['UNet']==True else self.layers[10](x10)
#         x12 = self.layers[11](x11)
#         x13 = self.layers[12](x12)
#         x14 = self.layers[13](x13) + x7 if self.hparams['UNet']==True else self.layers[13](x13)
#         x15 = self.layers[14](x14)
#         x16 = self.layers[15](x15) + x5 if self.hparams['UNet']==True else self.layers[15](x15)
#         x17 = self.layers[16](x16)
#         x18 = self.layers[17](x17) + x3 if self.hparams['UNet']==True else self.layers[17](x17)
#         x19 = self.layers[18](x18) + x1 if self.hparams['UNet']==True else self.layers[18](x18)
#         x20 = self.layers[19](x19)
#         x21 = self.layers[20](x20)
#         y = x21
#         return y
    
    
# class UNet(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self.hparams = params['hparams']
#         self.layers = nn.ModuleList()
#         self.num_channels = self.hparams['num_channels']
#         # Adjusted convolutional layers
#         self.layers.append(ReshapeLayer(output_shape=(1,self.hparams['h'],self.hparams['h'])))
#         self.layers.append(nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=4, stride=4, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
#         self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=2*self.num_channels, kernel_size=4, stride=4, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=2*self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
#         self.layers.append(nn.Conv2d(in_channels=2*self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=3, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=1, kernel_size=3, stride=3, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(ReshapeLayer(output_shape=(int(1*(self.hparams['h']/16)**2),)))
#         self.layers.append(nn.Linear(in_features=int(1*(self.hparams['h']/16)**2),out_features=int(1*(self.hparams['h']/16)**2)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(ReshapeLayer(output_shape=(1,int(self.hparams['h']/16),int(self.hparams['h']/16))))
#         self.layers.append(nn.ConvTranspose2d(1, self.num_channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
#         self.layers.append(nn.ConvTranspose2d(self.num_channels, 2*self.num_channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.BatchNorm2d(num_features=2*self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
#         self.layers.append(nn.ConvTranspose2d(2*self.num_channels, self.num_channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(nn.ReLU())
#         self.layers.append(nn.ConvTranspose2d(self.num_channels, 1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
#         self.layers.append(ReshapeLayer(output_shape=(self.hparams['h'],self.hparams['h'])))
#         if self.hparams['NLB_outputactivation'] is not None:
#             self.layers.append(self.hparams['NLB_outputactivation'])
            
#     def forward(self, x):
#         x0 = x
#         x1 = self.layers[0](x0)
#         x2 = self.layers[1](x1)
#         x3 = self.layers[2](x2)
#         x4 = self.layers[3](x3)
#         x5 = self.layers[4](x4)
#         x6 = self.layers[5](x5)
#         x7 = self.layers[6](x6)
#         x8 = self.layers[7](x7)
#         x9 = self.layers[8](x8)
#         x10 = self.layers[9](x9) 
#         x11 = self.layers[10](x10)
#         x12 = self.layers[11](x11)
#         x13 = self.layers[12](x12) + x11 if self.hparams['UNet']==True else self.layers[12](x12)
#         x14 = self.layers[13](x13)
#         x15 = self.layers[14](x14)
#         x16 = self.layers[15](x15) + x9 if self.hparams['UNet']==True else self.layers[15](x15)
#         x17 = self.layers[16](x16)
#         x18 = self.layers[17](x17)
#         x19 = self.layers[18](x18) + x6 if self.hparams['UNet']==True else self.layers[18](x18)
#         x20 = self.layers[19](x19)
#         x21 = self.layers[20](x20) 
#         x22 = self.layers[21](x21) + x3 if self.hparams['UNet']==True else self.layers[21](x21)
#         x23 = self.layers[22](x22) + x1 if self.hparams['UNet']==True else self.layers[22](x22)
#         x24 = self.layers[23](x23)
#         x25 = self.layers[24](x24)
#         y = x25
#         return y
    

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
    
    
class InvCNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = params['hparams']
        self.layers = nn.ModuleList()
        self.channels = 1
        
        # Adjusted convolutional layers
        self.layers.append(ReshapeLayer(output_shape=(1,self.hparams['h'],self.hparams['h'])))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=self.channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=self.channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=self.channels, out_channels=1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer(output_shape=(self.channels,int(self.hparams['h']/16),int(self.hparams['h']/16))))
        self.layers.append(nn.ReLU())
        self.layers.append(InversionLayer())
        self.layers.append(ReshapeLayer(output_shape=(self.channels, int(self.hparams['h']/16),int(self.hparams['h']/16))))
        self.layers.append(nn.ConvTranspose2d(1, self.channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=self.channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(self.channels, self.channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_features=self.channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(self.channels, self.channels, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(self.channels, 1, kernel_size=2, stride=2, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(ReshapeLayer(output_shape=(self.hparams['h'],self.hparams['h'])))
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
        self.kernel_size = 16
        self.num_channels = int(np.sqrt(20000/(2*self.kernel_size**2)))
        
        # Adjusted convolutional layers
        self.layers.append(ReshapeLayer(output_shape=(1,self.hparams['h'],self.hparams['h'])))
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, bias=self.hparams.get('bias_NLBranch', True)))
        
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, bias=self.hparams.get('bias_NLBranch', True)))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.BatchNorm2d(num_features=self.num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        
        self.layers.append(nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=1, kernel_size=self.kernel_size, stride=1, bias=self.hparams.get('bias_NLBranch', True)))
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