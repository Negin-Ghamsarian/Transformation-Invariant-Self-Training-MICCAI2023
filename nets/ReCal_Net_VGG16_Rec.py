#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 00:40:10 2022

@author: negin
"""

import torchvision.models as models
from torchsummary import summary
import torch.nn as nn
import torch
#from torchvision.ops import DeformConv2d
import torch.nn.functional as F
from .unet_parts import *


class Rec_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2, out_channels) 
        
    def forward(self,x):
        y = self.conv(self.up(x))
        
        return y
           
        
class Decoder_Unsup(nn.Module):
    def __init__(self, n_channels):
        super(Decoder_Unsup, self).__init__()
        self.up1 = Rec_Up(n_channels, 256)
        self.up2 = Rec_Up(256, 128)
        self.up3 = Rec_Up(128, 64)
        self.up4 = Rec_Up(64, 3)
        
    def forward(self,x):
        out1 = self.up1(x)
        out2 = self.up2(out1)
        out3 = self.up3(out2)
        out4 = self.up4(out3)
         
        return out4
    
    
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, size, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.norm = nn.LayerNorm([size,size], elementwise_affine=False)

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.relu2(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = self.norm(torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1)))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor




class RegionalSELayer(nn.Module):
    def __init__(self, num_channels,size):
        super(RegionalSELayer, self).__init__()
        
        
        self.avg1 = nn.AvgPool2d(3, stride=1, padding=1)
        self.avg2 = nn.AvgPool2d(5, stride=1, padding=2)
        self.avg3 = nn.AvgPool2d(7, stride=1, padding=3)
        
        self.conv0 = nn.Conv2d(num_channels, 1, 1)
        self.conv1 = nn.Conv2d(num_channels, 1, 1)
        self.conv2 = nn.Conv2d(num_channels, 1, 1)
        self.conv3 = nn.Conv2d(num_channels, 1, 1)
        
        self.fuse = nn.Conv2d(4, 1, 1)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm([size,size], elementwise_affine=False)
    def forward(self, x):
        
        batch_size, channel, a, b = x.size()
        
        y0 = self.conv0(x)
        y1 = self.conv1(self.avg1(x))
        y2 = self.conv2(self.avg2(x))
        y3 = self.conv3(self.avg3(x))
        
        concat = torch.cat((y0,y1,y2,y3), dim=1)
        #cmap = self.sigmoid(self.fuse(concat))
        cmap = self.relu(self.fuse(concat))
        cmap = cmap.view(batch_size, 1, a, b)
        
        output = torch.mul(x, cmap)
        
        return self.norm(output)
        
        
        


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, size, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, size, reduction_ratio)
        self.rSE = RegionalSELayer(num_channels, size)
        self.fc = nn.Conv2d(num_channels*2, num_channels, kernel_size=3, padding=1, groups=num_channels)
    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        
        y1 = self.cSE(input_tensor).unsqueeze(2)
        y2 = self.rSE(input_tensor).unsqueeze(2)
          
        y3 = torch.cat([y1, y2], dim = 2)
        y4 = torch.flatten(y3, start_dim=1, end_dim=2)
        #output_tensor = torch.max(self.cSE(input_tensor), self.rSE(input_tensor))
        output_tensor = self.fc(y4)
        return output_tensor





class VGG_Separate(nn.Module):
    def __init__(self):
        super(VGG_Separate,self).__init__()
        vgg_model = models.vgg16(pretrained=True)
        self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
        self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[4:9]) 
        self.Conv3 = nn.Sequential(*list(vgg_model.features.children())[9:16])
        self.Conv4 = nn.Sequential(*list(vgg_model.features.children())[16:23])
        self.Conv5 = nn.Sequential(*list(vgg_model.features.children())[23:30])

    def forward(self,x):
        out1 = self.Conv1(x)
        #print(out1.shape)
        out2 = self.Conv2(out1)
        #print(out2.shape)
        out3 = self.Conv3(out2)
        #print(out3.shape)
        out4 = self.Conv4(out3)
        #print(out4.shape)
        out5 = self.Conv5(out4)
        #print(out5.shape)

        return out1, out2, out3, out4, out5
        
                




class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dilations, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv1(in_channels, out_channels, dilations)
            

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)            


    


class DoubleConv1(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dilations):
        super().__init__()
           
        self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        
        self.out = nn.Sequential(
                   nn.BatchNorm2d(in_channels//2),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(in_channels//2, out_channels, kernel_size=3, padding=1),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True))
   

    def forward(self, x):
        
        x0 = self.conv(x)
        # weights = self.conv.weight
        # x1 = F.conv2d(x, weights, diltion = 3, padding=1)
        # x2 = F.conv2d(x, weights, diltion = 5, padding=1)

        
        # y = torch.cat([x0, x1, x2], dim=1)
        y1 = self.out(x0)
        
        return y1
        
        
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)        


class ReCal_Net_VGG16_Rec(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ReCal_Net_VGG16_Rec, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.Backbone = VGG_Separate()
        
        self.se1 = ChannelSpatialSELayer(512, 32)
        self.se2 = ChannelSpatialSELayer(256, 64)
        self.se3 = ChannelSpatialSELayer(128, 128)
        self.se4 = ChannelSpatialSELayer(64, 256)
        self.se5 = ChannelSpatialSELayer(32, 512)


        self.up1 = Up(1024, 256, [3, 6, 7], bilinear)
        self.up2 = Up(512, 128, [3, 6, 7], bilinear)
        self.up3 = Up(256, 64, [3, 6, 7], bilinear)
        self.up4 = Up(128, 32, [3, 6, 7], bilinear)

        self.outc = OutConv(32, n_classes)

        self.Rec_Branch = Decoder_Unsup(512)

    def forward(self, x, y):
        
        out5, out4, out3, out2, out1 = self.Backbone(x)



        out1 = self.se1(out1)
        x1 = self.se2(self.up1(out1, out2))
        x2 = self.se3(self.up2(x1, out3))
        x3 = self.se4(self.up3(x2, out4))
        x4 = self.se5(self.up4(x3, out5))
    
        

        
        logits = self.outc(x4)
        
        rec1, rec2, rec3, rec4, rec5 = self.Backbone(y) 
        #print(rec1.shape)
        rec_f = self.Rec_Branch(rec5)
        
        return logits, rec_f


    
    
    
if __name__ == '__main__':
    model = ReCal_Net_VGG16_Rec(n_channels=3, n_classes=3)
    summary(model, [(3, 512, 512), (3, 512, 512)])

    template = torch.ones((1, 3, 512, 512))
    template1 = torch.ones((1, 3, 512, 512))
    

    y1, y2 = model(template, template1)
    print(y2.shape) #[1, 10, 17, 17]  



