
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchsummary import summary


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, ksize = 3, stride = 2, pad = 0, dilation=1,
                 groups=1, inplace=False, has_bias=False):
        super(ConvBn, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        
        self.bn = nn.BatchNorm2d(out_channels)
        

    def forward(self, x):
        
        x = F.relu((self.bn(self.conv(x))))

        return x     


class FC_Drop(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FC_Drop, self).__init__()

        self.FC = nn.Linear(in_channels, out_channels)
        self.Drop = nn.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        
        x = self.FC(x)
        x = self.Drop(x)

        return x     


class Discriminator_Test_Time_UDA(nn.Module):
    def __init__(self):
        super(Discriminator_Test_Time_UDA, self).__init__()   

        self.conv1 = ConvBn(256+128+64+32, 256)
        self.conv2 = ConvBn(256, 128)  
        self.conv3 = ConvBn(128, 64)    
        self.conv4 = ConvBn(64, 32) 

        self.FC1 = FC_Drop(30752,256)
        self.FC2 = FC_Drop(256, 128)
        self.FC3 = FC_Drop(128, 1)

    def forward(self, x):

        x = self.conv1(x) 
        x = self.conv2(x)  
        x = self.conv3(x)   
        x = self.conv4(x) 

        # print(f'DDDDDDDDDDDsize{x.shape}')

        x = torch.flatten(x, start_dim=1)

        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)

        y = torch.sigmoid(x)
        
        return y


